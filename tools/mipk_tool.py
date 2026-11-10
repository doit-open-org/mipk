#!/usr/bin/env python3
"""Validate, create and inspect MIPK packages.

Modules are discovered from ops/, models/ and backends/ (each <module>/module.json);
more directories can be added with --module-dir."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "mipk.unified_pipeline.v2"
RUNTIME_VERSION = "2.0.0"
TASKS = {"vision_pipeline", "audio_pipeline"}
RESULT_TYPES = {
    "detection", "classification", "keypoint", "attribute", "track", "text"
}
INPUT_FORMATS = {"bgr", "nv12", "nv21", "pcm_s16le", "rgb"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIRS = [PROJECT_ROOT / "ops", PROJECT_ROOT / "models",
               PROJECT_ROOT / "backends"]

CONTAINER_MAGIC = b"MIPK3\0\0\0"
CONTAINER_VERSION = 3
HEADER_SIZE = 32
ENTRY_SIZE = 32
ENTRY_MODEL = 1
MAX_MANIFEST_SIZE = 1024 * 1024
MAX_ENTRY_COUNT = 64
HEADER = struct.Struct("<8sHHHHIIQ")
ENTRY = struct.Struct("<IIQQII")
SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class Capability:
    """What one module.json entry says about a task, backend, operator or postprocess."""

    def __init__(self, module: str, data: dict[str, Any]) -> None:
        self.module = module
        self.kind: str = data["kind"]
        self.name: str = data["name"]
        self.abi_version: int = data["abi_version"]
        self.implementation_version: str = data["implementation_version"]
        # The oldest module version a package built today should accept.
        self.compat_version: str = data.get("compat_version", data["implementation_version"])
        self.inputs: set[str] = set(data.get("inputs", []))
        raw_options = data.get("options", {})
        self.option_specs: dict[str, dict[str, Any]] = (
            raw_options if isinstance(raw_options, dict)
            else {name: {"type": "json"} for name in raw_options})
        self.options: set[str] = set(self.option_specs)
        self.constraints: list[dict[str, Any]] = data.get("constraints", [])
        self.output: str | None = data.get("output")
        self.collection_item: str | None = data.get("collection_item")
        self.collection_bound_option: str | None = data.get("collection_bound_option")
        result_types = data.get("result_types")
        self.result_types: set[str] | None = (
            set(result_types) if isinstance(result_types, list) else None)
        self.inherit_port: str | None = (
            result_types.split(":", 1)[1] if isinstance(result_types, str) and
            result_types.startswith("inherit:") else None)
        self.accepts: dict[str, set[str]] = {
            port: set(types) for port, types in data.get("accepts", {}).items()}
        self.stateful: bool = bool(data.get("stateful", False))
        self.result_bound_option: str = data.get("result_bound_option", "max_results")

    @property
    def identity(self) -> tuple[str, str]:
        return self.kind, self.name


def load_modules(extra_dirs: list[Path] | None = None) -> dict[tuple[str, str], Capability]:
    """Reads every module.json under the module directories."""
    capabilities: dict[tuple[str, str], Capability] = {}
    for base in MODULE_DIRS + list(extra_dirs or []):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/module.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data["capabilities"]:
                capability = Capability(data["module"], item)
                if capability.identity in capabilities:
                    raise RuntimeError(f"duplicate capability {capability.kind}:{capability.name}")
                capabilities[capability.identity] = capability
    return capabilities


CAPABILITIES = load_modules()


# ---------------------------------------------------------------------------
# Recipes: every model family ships a parameterised standard pipeline in
# models/<family>/recipe.json.  `new` renders one into a manifest so nobody
# has to write the graph by hand; the result is an ordinary manifest that can
# still be edited.
# ---------------------------------------------------------------------------

def _yolo_anchors(width: int, height: int) -> int:
    return sum((width // stride) * (height // stride) for stride in (8, 16, 32))


DERIVE_FUNCTIONS = {
    "yolo_anchors": _yolo_anchors,
    "yolo_result_types": lambda track: ["track" if track else "detection"],
    "pose_capacity": lambda max_results, keypoint_count: max_results * (1 + keypoint_count),
}


def derive_value(derive: dict[str, Any], values: dict[str, Any]) -> Any:
    """Evaluates a recipe's {"fn"|"expr", "args"} derivation from resolved parameters.

    "fn" names a built-in helper above; "expr" is an arithmetic expression over
    the named args, so a recipe outside this tree (mipk_extra_module) can derive
    values without the tool knowing about it."""
    args = derive.get("args", [])
    require(isinstance(args, list) and all(isinstance(arg, str) for arg in args),
            "derive.args must be a list of parameter names")
    if "expr" in derive:
        expr = derive["expr"]
        require(isinstance(expr, str) and re.fullmatch(r"[\w\s()+\-*/%.,<>=]+", expr) is not None
                and "__" not in expr,
                f"derive.expr is not a plain arithmetic expression: {expr!r}")
        names = set(re.findall(r"[A-Za-z_]\w*", expr)) - {"min", "max"}
        require(names <= set(args), f"derive.expr uses names outside args: "
                f"{', '.join(sorted(names - set(args)))}")
        return eval(expr, {"__builtins__": {}, "min": min, "max": max},  # noqa: S307
                    {arg: values[arg] for arg in args})
    fn = derive.get("fn")
    require(fn in DERIVE_FUNCTIONS, f"unknown derive function {fn!r}; use an expr instead")
    return DERIVE_FUNCTIONS[fn](*(values[arg] for arg in args))


class Recipe:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.name: str = data["name"]
        self.description: str = data.get("description", "")
        self.params: dict[str, dict[str, Any]] = data["params"]
        self.template: dict[str, Any] = data["manifest"]

    def resolve_params(self, given: dict[str, Any]) -> dict[str, Any]:
        """Applies defaults, derived values and type checks to the user's parameters."""
        unknown = set(given) - set(self.params)
        require(not unknown, f"recipe {self.name} has no parameter {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        pending = dict(self.params)
        # Defaults may reference other parameters ("${size}"), so resolve in passes.
        for _ in range(len(pending) + 1):
            for name, spec in list(pending.items()):
                if name in given:
                    value = given[name]
                elif "derive" in spec:
                    args = spec["derive"]["args"]
                    if any(arg not in values for arg in args):
                        continue
                    value = derive_value(spec["derive"], values)
                elif "default" in spec:
                    value = spec["default"]
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        ref = value[2:-1]
                        if ref not in values:
                            continue
                        value = values[ref]
                else:
                    require(not spec.get("required"), f"recipe {self.name} needs --set {name}=...")
                    value = None
                values[name] = self._coerce(name, spec, value)
                del pending[name]
        require(not pending, f"recipe {self.name}: unresolved parameters {', '.join(pending)}")
        return values

    @staticmethod
    def _coerce(name: str, spec: dict[str, Any], value: Any) -> Any:
        if value is None:
            return None
        kind = spec.get("type", "json")
        if kind == "int":
            if isinstance(value, str):
                value = int(value)
            require(type(value) is int, f"{name} must be an integer")
        elif kind == "number":
            if isinstance(value, str):
                value = float(value)
            require(_is_number(value), f"{name} must be a number")
        elif kind == "bool":
            if isinstance(value, str):
                require(value.lower() in {"true", "false", "1", "0"}, f"{name} must be true or false")
                value = value.lower() in {"true", "1"}
            require(type(value) is bool, f"{name} must be true or false")
        elif kind in {"string", "file"}:
            require(isinstance(value, str) and value, f"{name} must be a non-empty string")
            enum = spec.get("enum")
            require(enum is None or value in enum, f"{name} must be one of {', '.join(enum or [])}")
        elif kind == "labels":
            if isinstance(value, str):
                path = Path(value)
                if path.is_file():
                    # Stays a file reference; new/render copy the file alongside.
                    return {"file": path.name, "_source": str(path.resolve())}
                value = [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, dict) and "file" in value:
                return value
            # 元素可以是名字，也可以是 {name, enabled, score_threshold} 这类条目；
            # 条目本身由 validate_manifest 的 label_sets 校验
            require(isinstance(value, list) and value and
                    all(isinstance(item, (str, dict)) for item in value),
                    f"{name} must be a label file, a comma-separated list, or "
                    "a list of label entries")
        return value

    def render(self, given: dict[str, Any]) -> dict[str, Any]:
        values = self.resolve_params(given)
        placeholder = re.compile(r"\$\{([a-z_]+)\}")

        def substitute(node: Any) -> Any:
            if isinstance(node, dict):
                if "derive" in node and set(node) == {"derive"}:
                    return derive_value(node["derive"], values)
                condition = node.get("$if")
                if condition is not None and not values.get(condition):
                    return None
                out = {}
                for key, item in node.items():
                    if key == "$if":
                        continue
                    rendered = substitute(item)
                    if rendered is not None or not isinstance(item, dict):
                        out[key] = rendered
                return out
            if isinstance(node, list):
                return [item for item in (substitute(item) for item in node) if item is not None]
            if isinstance(node, str):
                match = placeholder.fullmatch(node)
                if match is not None:
                    require(match.group(1) in values, f"recipe references unknown parameter {match.group(1)}")
                    return values[match.group(1)]
                return placeholder.sub(lambda m: str(values[m.group(1)]), node)
            return node

        manifest = substitute(self.template)
        if manifest.get("output") == "$last":
            manifest["output"] = manifest["pipeline"][-1]["output"]
        return manifest


def load_recipes(extra_dirs: list[Path] | None = None) -> dict[str, Recipe]:
    recipes: dict[str, Recipe] = {}
    for base in MODULE_DIRS + list(extra_dirs or []):
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/recipe.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            recipes[data["name"]] = Recipe(path, data)
    return recipes


RECIPES = load_recipes()


def _strip_label_sources(manifest: dict[str, Any]) -> list[tuple[Path, str]]:
    """Removes the private _source markers, returning (source, name) pairs to copy."""
    files = []
    for labels in manifest.get("label_sets", {}).values():
        if isinstance(labels, dict) and "_source" in labels:
            files.append((Path(labels.pop("_source")), labels["file"]))
    return files


def render_recipe(name: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Renders in memory: label files are inlined so the result validates alone."""
    require(name in RECIPES, f"unknown recipe {name}; see `mipk_tool.py recipes`")
    manifest = RECIPES[name].render(params or {})
    for source, file_name in _strip_label_sources(manifest):
        for key, labels in manifest["label_sets"].items():
            if isinstance(labels, dict) and labels.get("file") == file_name:
                manifest["label_sets"][key] = [
                    line.strip() for line in source.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.startswith("#")]
    validate_manifest(manifest)
    return manifest


def parse_set(items: list[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in items:
        require("=" in item, f"--set expects name=value, got {item}")
        name, text = item.split("=", 1)
        try:
            values[name] = json.loads(text)
        except json.JSONDecodeError:
            values[name] = text
    return values


def compact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = copy.deepcopy(manifest)
    manifest.setdefault("version", "1.0.0")
    task = manifest.get("task", {})
    for key, default in (("result_capacity", 100), ("attachment_capacity", 0),
                         ("attachment_bytes", 0)):
        if task.get(key) == default:
            task.pop(key)
    task_type = "audio_pipeline" if manifest["input"]["format"] == "pcm_s16le" else "vision_pipeline"
    if task.get("type") == task_type:
        task.pop("type")
    if not task:
        manifest.pop("task", None)
    requires = manifest.get("requires", {})
    if requires.get("capabilities") == []:
        requires.pop("capabilities")
    if not requires:
        manifest.pop("requires", None)
    fields = ("format_version", "schema", "name", "version", "requires", "task",
              "models", "input", "label_sets", "pipeline", "output")
    manifest = {key: manifest[key] for key in fields if key in manifest}
    for stage in manifest["pipeline"]:
        fields = [key for key in ("id", "type", "operator", "model", "postprocess", "label_set")
                  if key in stage]
        fields += [key for key in stage if key not in fields and
                   key not in {"execution", "inputs", "output"}]
        fields += [key for key in ("execution", "inputs", "output") if key in stage]
        ordered = {key: stage[key] for key in fields}
        stage.clear()
        stage.update(ordered)
    return manifest


def write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    (output / "manifest.json").write_text(
        json.dumps(compact_manifest(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def new_recipe(name: str, output: Path, params: dict[str, Any], link: bool) -> None:
    """Creates a recipe directory: recipe.json (the source), model and label
    files beside it, and manifest.json rendered from them."""
    require(name in RECIPES, f"unknown recipe {name}; see `mipk_tool.py recipes`")
    recipe = RECIPES[name]
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "recipe.json").exists() and not (output / "manifest.json").exists(),
            f"{output} already holds a recipe or manifest")
    files: list[tuple[Path, str]] = []
    compact: dict[str, Any] = {"recipe": name}
    for param, value in params.items():
        spec = recipe.params.get(param, {})
        if spec.get("type") == "file":
            source = Path(value)
            require(source.is_file(), f"{param}: {source} is not a file")
            files.append((source, source.name))
            value = source.name
        elif spec.get("type") == "labels" and isinstance(value, str) and Path(value).is_file():
            source = Path(value)
            files.append((source, source.name))
            value = source.name
        compact[param] = value
    for source, target_name in files:
        target = output / target_name
        if target.exists() and target.resolve() == source.resolve():
            continue
        if link:
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)
    (output / "recipe.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    render_dir(output)


def render_dir(directory: Path) -> None:
    """Rewrites manifest.json from recipe.json after the latter was edited."""
    recipe_file = directory / "recipe.json"
    require(recipe_file.is_file(), f"{directory} has no recipe.json; edit manifest.json directly")
    manifest = render_recipe_file(recipe_file)
    _strip_label_sources(manifest)
    validate_manifest(manifest, directory)
    write_manifest(directory, manifest)


def capability(kind: str, name: str) -> Capability | None:
    return CAPABILITIES.get((kind, name))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_keys(value: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = set(value) - allowed
    require(not unknown, f"{field} contains unknown fields: {', '.join(sorted(unknown))}")


def positive_int(value: Any, field: str) -> int:
    require(type(value) is int and value > 0, f"{field} must be a positive integer")
    return value


def nonnegative_int(value: Any, field: str) -> int:
    require(type(value) is int and value >= 0,
            f"{field} must be a non-negative integer")
    return value


def score(value: Any, field: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"{field} must be in [0, 1]")
    return float(value)


def semver(value: Any, field: str) -> tuple[int, int, int]:
    require(isinstance(value, str), f"{field} must be a semantic version")
    match = SEMVER.fullmatch(value)
    require(match is not None, f"{field} must use MAJOR.MINOR.PATCH")
    parts = tuple(int(item) for item in match.groups())
    require(all(item <= 0xFFFFFFFF for item in parts), f"{field} overflows uint32")
    return parts  # type: ignore[return-value]


def relative_path(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a non-empty string")
    require("\\" not in value and not value.startswith("/"), f"{field} is unsafe")
    require(all(part not in {"", ".", ".."} for part in value.split("/")),
            f"{field} is unsafe")
    return Path(value)


def validate_label_sets(value: Any) -> dict[str, list[Any]]:
    require(isinstance(value, dict) and value, "label_sets must be a non-empty object")
    for set_name, labels in value.items():
        require(isinstance(set_name, str) and set_name,
                "label set names must not be empty")
        require(isinstance(labels, list) and labels,
                f"label_sets.{set_name} must be a non-empty array")
        names: set[str] = set()
        source_ids: set[int] = set()
        for index, item in enumerate(labels):
            if isinstance(item, str):
                name, source_id = item, index
            else:
                require(isinstance(item, dict),
                        f"label_sets.{set_name}[{index}] must be a string or object")
                exact_keys(item, {"name", "source_id", "enabled", "score_threshold"},
                           f"label_sets.{set_name}[{index}]")
                name = item.get("name")
                source_id = item.get("source_id", index)
                require(type(item.get("enabled", True)) is bool,
                        f"label_sets.{set_name}[{index}].enabled must be boolean")
                if "score_threshold" in item:
                    score(item["score_threshold"],
                          f"label_sets.{set_name}[{index}].score_threshold")
            require(isinstance(name, str) and name,
                    f"label_sets.{set_name}[{index}].name is required")
            require(name not in names, f"label_sets.{set_name} names must be unique")
            names.add(name)
            require(source_id is None or type(source_id) is int and 0 <= source_id <= 0x7FFFFFFF,
                    f"label_sets.{set_name}[{index}].source_id is invalid")
            if source_id is not None:
                require(source_id not in source_ids,
                        f"label_sets.{set_name} source IDs must be unique")
                source_ids.add(source_id)
    return value


def validate_requirement(item: Any, index: int) -> tuple[str, str]:
    require(isinstance(item, dict), f"requires.capabilities[{index}] must be an object")
    exact_keys(item, {"kind", "name", "abi_version", "min_implementation_version"},
               f"requires.capabilities[{index}]")
    kind, name = item.get("kind"), item.get("name")
    require(kind in {"backend", "operator", "postprocess"} and
            isinstance(name, str) and name,
            f"requires.capabilities[{index}] has invalid identity")
    positive_int(item.get("abi_version"), f"requires.capabilities[{index}].abi_version")
    semver(item.get("min_implementation_version"),
           f"requires.capabilities[{index}].min_implementation_version")
    return kind, name


def validate_execution(value: Any, index: int) -> None:
    if value is None:
        return
    require(isinstance(value, dict), f"pipeline[{index}].execution must be an object")
    exact_keys(value, {"mode", "max_items"}, f"pipeline[{index}].execution")
    mode = value.get("mode")
    require(mode in {"once", "per_item"},
            f"pipeline[{index}].execution.mode is invalid")
    if mode == "once":
        require("max_items" not in value,
                f"pipeline[{index}].execution.max_items requires per_item")
    else:
        positive_int(value.get("max_items"),
                     f"pipeline[{index}].execution.max_items")


def binding_source(value: Any, field: str) -> tuple[str, int | None]:
    if isinstance(value, str):
        return value, None
    require(isinstance(value, dict), f"{field} must be a value name or tensor binding")
    exact_keys(value, {"source", "tensor_index"}, field)
    source = value.get("source")
    require(isinstance(source, str) and source, f"{field}.source is required")
    tensor_index = nonnegative_int(value.get("tensor_index"), f"{field}.tensor_index")
    return source, tensor_index


def _is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _check_scalar(value: Any, spec: dict[str, Any], field: str,
                  result_capacity: int) -> None:
    kind = spec.get("type", "json")
    if kind == "bool":
        require(type(value) is bool, f"{field} must be boolean")
        return
    if kind == "string":
        require(isinstance(value, str), f"{field} must be a string")
        enum = spec.get("enum")
        require(enum is None or value in enum,
                f"{field} must be one of {', '.join(enum or [])}")
        return
    if kind == "int":
        require(type(value) is int, f"{field} must be an integer")
    elif kind == "number":
        require(_is_number(value), f"{field} must be a finite number")
    else:
        return
    minimum, maximum = spec.get("min"), spec.get("max")
    if maximum == "result_capacity":
        maximum = result_capacity
    require(minimum is None or value >= minimum, f"{field} must be at least {minimum}")
    require(maximum is None or value <= maximum, f"{field} must be at most {maximum}")
    exclusive = spec.get("exclusive_min")
    require(exclusive is None or value > exclusive,
            f"{field} must be greater than {exclusive}")


def _check_option(value: Any, spec: dict[str, Any], field: str,
                  result_capacity: int) -> None:
    kind = spec.get("type", "json")
    if kind == "number_or_array":
        if _is_number(value):
            return
        require(isinstance(value, list) and len(value) == spec.get("length", len(value))
                and all(_is_number(item) for item in value),
                f"{field} must be a number or an array of {spec.get('length')} numbers")
        return
    if kind == "array":
        require(isinstance(value, list), f"{field} must be an array")
        length = spec.get("length")
        require(length is None or len(value) == length,
                f"{field} must contain {length} items")
        items = spec.get("items", "json")
        for position, item in enumerate(value):
            if items == "point":
                require(isinstance(item, list) and len(item) == 2 and
                        all(_is_number(axis) for axis in item),
                        f"{field}[{position}] must be an [x, y] point")
            else:
                _check_scalar(item, {"type": items}, f"{field}[{position}]",
                              result_capacity)
        if spec.get("unique"):
            hashable = [json.dumps(item, sort_keys=True) for item in value]
            require(len(set(hashable)) == len(hashable), f"{field} items must be unique")
        return
    _check_scalar(value, spec, field, result_capacity)


def _check_constraint(constraint: dict[str, Any], stage: dict[str, Any],
                      field: str) -> None:
    if "together" in constraint:
        present = [name in stage for name in constraint["together"]]
        require(all(present) or not any(present),
                f"{field} options {', '.join(constraint['together'])} go together")
    elif "order" in constraint:
        values = [stage.get(name) for name in constraint["order"]]
        if all(_is_number(value) for value in values):
            require(all(a <= b for a, b in zip(values, values[1:])),
                    f"{field} options {', '.join(constraint['order'])} must be non-decreasing")
    elif "le" in constraint:
        left, right = (stage.get(name) for name in constraint["le"])
        if _is_number(left) and _is_number(right):
            require(left <= right, f"{field}.{constraint['le'][0]} exceeds {constraint['le'][1]}")
    else:
        raise ValueError(f"{field} has an unknown constraint {constraint}")


def validate_node_options(stage: dict[str, Any], index: int,
                          result_capacity: int) -> None:
    """Checks a stage's options against its module's declared specs.

    Types, ranges, enums and the relations in `constraints` all come from
    module.json, so a third-party module gets the same depth of checking as a
    bundled one. NODE_VALIDATORS holds the few geometric checks that have no
    declarative form."""
    kind = stage["type"]
    name = stage.get("operator") or stage.get("postprocess")
    common = {"id", "type", "inputs", "output", "execution"}
    if kind == "model":
        exact_keys(stage, common | {"model", "metadata_from"}, f"pipeline[{index}]")
        return
    node = capability(kind, name)
    require(node is not None, f"pipeline[{index}] {kind} '{name}' is not a known module; "
            "add its module directory with --module-dir")
    allowed = common | {kind} | node.options
    if kind == "postprocess":
        allowed |= {"label_set"}
    exact_keys(stage, allowed, f"pipeline[{index}]")
    for option, spec in node.option_specs.items():
        field = f"pipeline[{index}].{option}"
        if option not in stage:
            require(not spec.get("required"), f"{field} is required")
            continue
        _check_option(stage[option], spec, field, result_capacity)
    for constraint in node.constraints:
        _check_constraint(constraint, stage, f"pipeline[{index}]")
    extra = NODE_VALIDATORS.get(name)
    if extra is not None:
        extra(stage, index)


def _validate_image_preprocess(stage: dict[str, Any], index: int) -> None:
    require(stage.get("format", "auto") in {"auto", "nv12", "float"},
            f"pipeline[{index}].format is invalid")
    require(stage.get("interpolation", "bilinear") in {"bilinear", "nearest"},
            f"pipeline[{index}].interpolation is invalid")


def _validate_image_align(stage: dict[str, Any], index: int) -> None:
    target_points = stage.get("target_points")
    if not isinstance(target_points, list) or len(target_points) != 3:
        return
    area = ((target_points[1][0] - target_points[0][0]) *
            (target_points[2][1] - target_points[0][1]) -
            (target_points[1][1] - target_points[0][1]) *
            (target_points[2][0] - target_points[0][0]))
    require(abs(area) > 1.0e-9, f"pipeline[{index}].target_points must not be collinear")
    width, height = stage.get("output_width"), stage.get("output_height")
    if _is_number(width) and _is_number(height):
        require(all(0 <= point[0] <= width and 0 <= point[1] <= height
                    for point in target_points),
                f"pipeline[{index}].target_points must be inside the output")


# Geometry that module.json cannot express. Everything else is declarative.
NODE_VALIDATORS = {
    "image_preprocess": _validate_image_preprocess,
    "image_align": _validate_image_align,
}


def normalize_manifest(data: Any, base_dir: Path | None = None) -> dict[str, Any]:
    """Fills in everything the runtime defaults, returning a complete manifest.

    name, version, requires, task and stage ids are optional in a hand-written
    manifest; a label set may be {"file": "labels.txt"} beside the manifest.
    The packaging tool always packs the normalized form."""
    require(isinstance(data, dict), "manifest root must be an object")
    data = copy.deepcopy(data)
    data.setdefault("name", "pipeline")
    data.setdefault("version", "1.0.0")
    requires = data.setdefault("requires", {})
    if isinstance(requires, dict):
        requires.setdefault("runtime_min_version", RUNTIME_VERSION)
        requires.setdefault("capabilities", [])
    task = data.setdefault("task", {})
    if isinstance(task, dict):
        task.setdefault("result_capacity", 100)
        task.setdefault("attachment_capacity", 0)
        task.setdefault("attachment_bytes", 0)
    pipeline = data.get("pipeline")
    if isinstance(pipeline, list):
        for stage in pipeline:
            if isinstance(stage, dict) and "id" not in stage and isinstance(stage.get("output"), str):
                stage["id"] = stage["output"]
    label_sets = data.get("label_sets")
    if isinstance(label_sets, dict):
        for name, labels in list(label_sets.items()):
            if isinstance(labels, dict) and set(labels) == {"file"}:
                require(base_dir is not None, f"label_sets.{name}.file needs the manifest's directory")
                path = base_dir / relative_path(labels["file"], f"label_sets.{name}.file")
                require(path.is_file(), f"label file not found: {path}")
                label_sets[name] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                                    if line.strip() and not line.startswith("#")]
    return data


def validate_manifest(data: Any, base_dir: Path | None = None) -> list[Path]:
    data = normalize_manifest(data, base_dir)
    exact_keys(data, {"format_version", "schema", "name", "version", "requires",
                      "task", "models", "input", "label_sets", "pipeline", "output"}, "$")
    require(data.get("format_version") == 2, "format_version must be 2")
    require(data.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    require(isinstance(data.get("name"), str) and data["name"], "name is required")
    semver(data.get("version"), "version")
    requires = data.get("requires")
    require(isinstance(requires, dict), "requires must be an object")
    exact_keys(requires, {"runtime_min_version", "capabilities"}, "requires")
    require(semver(requires.get("runtime_min_version"), "requires.runtime_min_version") <=
            semver(RUNTIME_VERSION, "runtime"), "runtime_min_version is unsupported")
    capability_items = requires.get("capabilities")
    require(isinstance(capability_items, list), "requires.capabilities must be an array")
    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(capability_items):
        identity = validate_requirement(item, index)
        require(identity not in declared, "capability requirements must be unique")
        declared[identity] = item

    task = data.get("task")
    require(isinstance(task, dict), "task must be an object")
    exact_keys(task, {"type", "result_types", "result_capacity",
                      "attachment_capacity", "attachment_bytes"}, "task")
    task_type = task.get("type")
    require(task_type is None or task_type in TASKS,
            "task.type must be vision_pipeline or audio_pipeline when given")
    result_types = task.get("result_types", sorted(RESULT_TYPES))
    require(isinstance(result_types, list) and result_types and
            len(result_types) == len(set(result_types)) and
            set(result_types) <= RESULT_TYPES, "task.result_types is invalid")
    result_capacity = positive_int(task.get("result_capacity"), "task.result_capacity")
    require(nonnegative_int(task.get("attachment_capacity"),
                            "task.attachment_capacity") == 0 and
            nonnegative_int(task.get("attachment_bytes"), "task.attachment_bytes") == 0,
            "stage 1 does not support attachments")

    label_sets = validate_label_sets(data.get("label_sets"))
    models = data.get("models")
    require(isinstance(models, list) and 1 <= len(models) <= MAX_ENTRY_COUNT,
            f"models must contain 1 to {MAX_ENTRY_COUNT} entries")
    model_ids: set[str] = set()
    model_paths: list[Path] = []
    for index, model in enumerate(models):
        require(isinstance(model, dict), f"models[{index}] must be an object")
        exact_keys(model, {"id", "backend", "file", "kind"}, f"models[{index}]")
        model_id = model.get("id")
        require(isinstance(model_id, str) and model_id and model_id not in model_ids,
                "models[].id must be non-empty and unique")
        model_ids.add(model_id)
        backend = model.get("backend")
        require(isinstance(backend, str) and backend, f"models[{index}].backend is required")
        require(capability("backend", backend) is not None,
                f"models[{index}].backend '{backend}' is not a known module")
        model_paths.append(relative_path(model.get("file"), f"models[{index}].file"))

    input_info = data.get("input")
    require(isinstance(input_info, dict) and input_info.get("format") in INPUT_FORMATS,
            "input.format is unsupported")
    if input_info["format"] == "pcm_s16le":
        exact_keys(input_info, {"format", "sample_rate", "channels", "frame_samples",
                                "reset_gap_ms"}, "input")
        for key in ("sample_rate", "channels", "frame_samples", "reset_gap_ms"):
            positive_int(input_info.get(key), f"input.{key}")
        require(task_type in (None, "audio_pipeline"), "pcm_s16le is an audio_pipeline")
    else:
        exact_keys(input_info, {"format", "width", "height"}, "input")
        positive_int(input_info.get("width"), "input.width")
        positive_int(input_info.get("height"), "input.height")
        require(task_type in (None, "vision_pipeline"), "image input is a vision_pipeline")

    pipeline = data.get("pipeline")
    require(isinstance(pipeline, list) and 1 <= len(pipeline) <= 32,
            "pipeline must contain 1 to 32 stages")
    available = {"$input"}
    node_ids: set[str] = set()
    value_result_types: dict[str, set[str] | None] = {"$input": None}
    value_is_collection = {"$input": False}
    value_max_items: dict[str, int | None] = {"$input": None}
    model_references = {model_id: 0 for model_id in model_ids}
    used_capabilities: set[tuple[str, str]] = {
        ("backend", model["backend"]) for model in models}
    stage_sources: list[set[str]] = []
    stage_payloads: list[str] = []
    stage_per_item_max: list[int | None] = []
    for index, stage in enumerate(pipeline):
        require(isinstance(stage, dict), f"pipeline[{index}] must be an object")
        node_id = stage.get("id")
        require(isinstance(node_id, str) and node_id and node_id not in node_ids,
                f"pipeline[{index}].id must be non-empty and unique")
        node_ids.add(node_id)
        kind = stage.get("type")
        stage_inputs = stage.get("inputs")
        if kind == "model":
            name = stage.get("model")
            require(name in model_ids, f"pipeline[{index}].model is invalid")
            model_references[name] += 1
            inferred_results: set[str] | None = None
            output_payload = "tensors"
        elif kind in {"operator", "postprocess"}:
            field = kind
            name = stage.get(field)
            require(isinstance(name, str) and name, f"pipeline[{index}].{field} is required")
            node = capability(kind, name)
            require(node is not None,
                    f"pipeline[{index}] {kind} '{name}' is not a known module")
            used_capabilities.add((kind, name))
            if kind == "postprocess":
                require(stage.get("label_set") in label_sets,
                        f"pipeline[{index}].label_set is invalid")
            bindings_for_types = stage_inputs if isinstance(stage_inputs, dict) else {}
            for port, accepted in node.accepts.items():
                binding = bindings_for_types.get(port)
                source = binding_source(binding, f"pipeline[{index}].inputs.{port}")[0]
                actual = value_result_types.get(source)
                require(actual is not None and actual <= accepted,
                        f"pipeline[{index}] {name} input '{port}' accepts only "
                        f"{', '.join(sorted(accepted))} results")
            if node.output == "results":
                if node.inherit_port is not None:
                    binding = bindings_for_types.get(node.inherit_port)
                    source = binding_source(
                        binding, f"pipeline[{index}].inputs.{node.inherit_port}")[0]
                    inferred_results = value_result_types.get(source)
                    require(inferred_results is not None,
                            f"pipeline[{index}] requires result input '{node.inherit_port}'")
                    inferred_results = set(inferred_results)
                else:
                    require(node.result_types is not None,
                            f"pipeline[{index}] {name} declares no result types")
                    inferred_results = set(node.result_types)
                output_payload = "results"
            elif node.output == "collection":
                inferred_results = None
                output_payload = node.collection_item or "image_view"
            elif node.output in {"image_view", "tensors"}:
                inferred_results = None
                output_payload = node.output
            else:
                raise ValueError(f"pipeline[{index}] {name} declares no output kind")
        else:
            raise ValueError(f"pipeline[{index}].type is invalid")
        bindings = stage_inputs
        require(isinstance(bindings, dict) and 1 <= len(bindings) <= 8 and
                all(isinstance(port, str) and port for port in bindings),
                f"pipeline[{index}].inputs has an invalid binding")
        resolved_bindings = {
            port: binding_source(binding, f"pipeline[{index}].inputs.{port}")
            for port, binding in bindings.items()
        }
        require(all(source in available for source, _ in resolved_bindings.values()),
                f"pipeline[{index}].inputs references an unavailable value")
        validate_execution(stage.get("execution"), index)
        execution = stage.get("execution")
        per_item_max = (execution["max_items"] if isinstance(execution, dict) and
                        execution.get("mode") == "per_item" else None)
        collection_sources = [source for source, _ in resolved_bindings.values()
                              if value_is_collection[source]]
        if per_item_max is not None:
            require(bool(collection_sources),
                    f"pipeline[{index}] per_item requires a collection input")
            require(kind == "model" or not capability(kind, name).stateful,
                    f"pipeline[{index}] stateful operator cannot use per_item")
            for source in collection_sources:
                source_max = value_max_items[source]
                require(source_max is not None and source_max <= per_item_max,
                        f"pipeline[{index}].execution.max_items is smaller than {source}")
        else:
            require(not collection_sources,
                    f"pipeline[{index}] collection input requires per_item execution")
        if kind == "model":
            require(all(port.isdecimal() for port in bindings),
                    f"pipeline[{index}] model ports must be tensor indices")
        else:
            require(all(tensor_index is None for _, tensor_index in resolved_bindings.values()),
                    f"pipeline[{index}] tensor_index is only valid on model inputs")
            node_inputs = capability(kind, name).inputs
            if node_inputs:
                require(set(bindings) == node_inputs,
                        f"pipeline[{index}] inputs do not match {name} schema")
        output = stage.get("output")
        require(isinstance(output, str) and output and output not in available and
                output != "$input", f"pipeline[{index}].output must be unique")
        validate_node_options(stage, index, result_capacity)
        available.add(output)
        value_result_types[output] = inferred_results
        is_collection_node = (kind != "model" and
                              capability(kind, name).output == "collection")
        value_is_collection[output] = per_item_max is not None or is_collection_node
        value_max_items[output] = (per_item_max if per_item_max is not None else
                                   stage.get(capability(kind, name).collection_bound_option
                                             or "max_items")
                                   if is_collection_node else None)
        stage_sources.append({source for source, _ in resolved_bindings.values()})
        stage_payloads.append(output_payload)
        stage_per_item_max.append(per_item_max)
    require(all(count == 1 for count in model_references.values()),
            "pipeline must reference every model exactly once")
    for index, maximum in enumerate(stage_per_item_max):
        if maximum is None:
            continue
        continues = (index + 1 < len(pipeline) and
                     stage_per_item_max[index + 1] == maximum)
        consumers = [consumer for consumer, sources in enumerate(stage_sources)
                     if consumer > index and pipeline[index]["output"] in sources]
        if continues:
            require(consumers == [index + 1],
                    f"pipeline[{index}] fusion output must have one next-stage consumer")
        else:
            require(stage_payloads[index] != "tensors",
                    f"pipeline[{index}] tensor collection cannot end a fusion group")
    selected_output = data.get("output")
    require(selected_output in available and selected_output != "$input",
            "output must select a stage output")
    output_result_types = value_result_types.get(selected_output)
    require(output_result_types is not None and output_result_types <= set(result_types),
            "task.result_types does not cover the pipeline output")
    require(all(identity in used_capabilities for identity in declared),
            "requires.capabilities contains an unused capability")
    for identity, requirement in declared.items():
        node = capability(*identity)
        require(node is not None, f"capability {identity[0]}/{identity[1]} is unknown")
        require(requirement["abi_version"] == node.abi_version,
                f"capability {identity[0]}/{identity[1]} has unsupported ABI")
    for stage in pipeline:
        for field in ("max_results", "max_items", "max_tracks"):
            if field in stage:
                require(stage[field] <= result_capacity,
                        f"{field} exceeds task.result_capacity")
        execution = stage.get("execution")
        if isinstance(execution, dict) and execution.get("mode") == "per_item":
            require(execution["max_items"] <= result_capacity,
                    "execution.max_items exceeds task.result_capacity")
    return model_paths


def load_manifest(path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Loads and validates a manifest; the returned data is normalized."""
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    paths = validate_manifest(data, path.parent)
    return normalize_manifest(data, path.parent), paths


def load_recipe_dir(directory: Path) -> tuple[dict[str, Any], list[Path]]:
    """A directory holds either recipe.json (rendered fresh) or manifest.json."""
    recipe_file = directory / "recipe.json"
    if recipe_file.is_file():
        data = render_recipe_file(recipe_file)
        _strip_label_sources(data)
        manifest_file = directory / "manifest.json"
        if manifest_file.is_file():
            try:
                current = json.loads(manifest_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = None
            if current != compact_manifest(data):
                print(f"note: {manifest_file} differs from recipe.json; "
                      "run `mipk_tool.py render` to refresh it", file=sys.stderr)
        paths = validate_manifest(data, directory)
        return normalize_manifest(data, directory), paths
    return load_manifest(directory / "manifest.json")


def render_recipe_file(recipe_file: Path) -> dict[str, Any]:
    """recipe.json: {"recipe": "yolo", "model": "x.onnx", "labels": "l.txt", ...params}."""
    spec = json.loads(recipe_file.read_text(encoding="utf-8"))
    require(isinstance(spec, dict) and isinstance(spec.get("recipe"), str),
            f"{recipe_file} must name a recipe")
    params = {key: value for key, value in spec.items() if key not in {"recipe", "name"}}
    recipe = RECIPES.get(spec["recipe"])
    require(recipe is not None, f"unknown recipe {spec['recipe']}")
    directory = recipe_file.parent
    for name, param_spec in recipe.params.items():
        if name not in params:
            continue
        if param_spec.get("type") == "labels" and isinstance(params[name], str):
            candidate = directory / params[name]
            if candidate.is_file():
                params[name] = str(candidate)
    manifest = recipe.render(params)
    # The package is named after the directory unless recipe.json says otherwise.
    manifest["name"] = spec.get("name", directory.resolve().name)
    require(isinstance(manifest["name"], str) and manifest["name"], "name must be a string")
    return manifest


def pin_requirements(data: dict[str, Any]) -> dict[str, Any]:
    """Fills requires.capabilities with every module the pipeline uses."""
    data = normalize_manifest(data)
    identities: list[tuple[str, str]] = []
    identities += [("backend", model["backend"]) for model in data["models"]]
    for stage in data["pipeline"]:
        if stage["type"] in {"operator", "postprocess"}:
            identities.append((stage["type"], stage[stage["type"]]))
    declared = {(item["kind"], item["name"]) for item in data["requires"]["capabilities"]}
    for identity in sorted(set(identities)):
        if identity in declared:
            continue
        node = capability(*identity)
        data["requires"]["capabilities"].append({
            "kind": node.kind, "name": node.name,
            "abi_version": node.abi_version,
            "min_implementation_version": node.compat_version,
        })
    return data


def crc32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def create_package(manifest: Path, model: Path | None, output: Path,
                   pin: bool, backend: str | None = None) -> None:
    require(manifest.is_file(), f"manifest not found: {manifest}")
    require(output.suffix == ".mipk" and not output.exists(),
            "output must be a new .mipk file")
    manifest_data, paths = (load_recipe_dir(manifest.parent) if manifest.name == "manifest.json"
                            else load_manifest(manifest))
    if backend is not None:
        # The recipe is chip-neutral; the backend and model file are not.
        require(capability("backend", backend) is not None,
                f"backend '{backend}' is not a known module")
        for item in manifest_data["models"]:
            item["backend"] = backend
        validate_manifest(manifest_data)
    if pin:
        manifest_data = pin_requirements(manifest_data)
        validate_manifest(manifest_data)
    if model is not None:
        require(len(paths) == 1, "--model only supports one model")
        paths = [model]
    else:
        paths = [manifest.parent / path for path in paths]
    require(all(path.is_file() for path in paths), "one or more model files are missing")
    manifest_bytes = (json.dumps(manifest_data, ensure_ascii=False,
                                 separators=(",", ":")) + "\n").encode()
    require(len(manifest_bytes) <= MAX_MANIFEST_SIZE, "manifest is too large")
    payloads = [path.read_bytes() for path in paths]
    require(all(payloads), "models must not be empty")
    entries = []
    offset = HEADER_SIZE + ENTRY_SIZE * len(payloads) + len(manifest_bytes)
    for logical_index, payload in enumerate(payloads):
        entries.append(ENTRY.pack(ENTRY_MODEL, 0, offset, len(payload),
                                  crc32(payload), logical_index))
        offset += len(payload)
    header = HEADER.pack(CONTAINER_MAGIC, CONTAINER_VERSION, HEADER_SIZE, 0, 0,
                         len(manifest_bytes), len(entries), offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{output.name}.",
                                         dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(header)
            stream.write(b"".join(entries))
            stream.write(manifest_bytes)
            for payload in payloads:
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def read_package(package: Path) -> dict[str, Any]:
    """Validates a .mipk file, a recipe directory or a bare manifest.json."""
    if package.suffix == ".json":
        return load_manifest(package)[0]
    if package.is_dir():
        data, paths = load_recipe_dir(package)
        for path in paths:
            require((package / path).is_file(), f"model file missing: {path}")
        return data
    require(package.suffix == ".mipk" and package.is_file(), "package must be a .mipk file")
    content = package.read_bytes()
    require(len(content) >= HEADER_SIZE, "package is truncated")
    magic, version, header_size, flags, _, manifest_size, count, package_size = (
        HEADER.unpack_from(content))
    require((magic, version, header_size, flags) ==
            (CONTAINER_MAGIC, CONTAINER_VERSION, HEADER_SIZE, 0),
            "unsupported container header")
    require(package_size == len(content) and 0 < manifest_size <= MAX_MANIFEST_SIZE and
            0 < count <= MAX_ENTRY_COUNT, "invalid package size")
    manifest_offset = HEADER_SIZE + count * ENTRY_SIZE
    expected_offset = manifest_offset + manifest_size
    for index in range(count):
        entry_type, entry_flags, offset, size, checksum, logical = ENTRY.unpack_from(
            content, HEADER_SIZE + index * ENTRY_SIZE)
        require(entry_type == ENTRY_MODEL and entry_flags == 0 and logical == index and
                offset == expected_offset and size > 0 and
                offset + size <= len(content), "invalid model entry")
        require(crc32(content[offset:offset + size]) == checksum,
                f"model[{index}] checksum mismatch")
        expected_offset += size
    require(expected_offset == len(content), "package has trailing payload")
    try:
        data = json.loads(content[manifest_offset:manifest_offset + manifest_size].decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid manifest") from error
    validate_manifest(data)
    require(count == len(data["models"]), "model entries do not match manifest")
    return data


def validate_target(data: dict[str, Any], target: Path | None) -> None:
    if target is None:
        return
    catalog = json.loads(target.read_text(encoding="utf-8"))
    entries = catalog.get("capabilities") if isinstance(catalog, dict) else catalog
    require(isinstance(entries, list), "target capability catalog must contain an array")
    available: dict[tuple[str, str], dict[str, Any]] = {}
    for index, item in enumerate(entries):
        require(isinstance(item, dict),
                f"target capabilities[{index}] must be an object")
        kind = item.get("kind")
        name = item.get("name")
        require(isinstance(kind, str) and isinstance(name, str) and bool(name),
                f"target capabilities[{index}] has invalid identity")
        identity = (kind, name)
        require(kind in {"backend", "operator", "postprocess"},
                f"target capabilities[{index}] has invalid identity")
        require(identity not in available,
                "target capability identities must be unique")
        positive_int(item.get("abi_version"),
                     f"target capabilities[{index}].abi_version")
        semver(item.get("implementation_version"),
               f"target capabilities[{index}].implementation_version")
        available[identity] = item
    identities = [("backend", model["backend"]) for model in data["models"]]
    for stage in data["pipeline"]:
        if stage["type"] in {"operator", "postprocess"}:
            identities.append((stage["type"], stage[stage["type"]]))
    requirements = {(item["kind"], item["name"]): item
                    for item in data.get("requires", {}).get("capabilities", [])}
    for identity in set(identities):
        require(identity in available, f"target is missing {identity[0]}/{identity[1]}")
        requirement = requirements.get(identity)
        if requirement is not None:
            required_abi = requirement["abi_version"]
            minimum_version = requirement["min_implementation_version"]
        else:
            node = capability(*identity)
            required_abi = node.abi_version
            minimum_version = "0.0.0"
        actual = available[identity]
        require(actual["abi_version"] == required_abi,
                f"target ABI mismatch for {identity[0]}/{identity[1]}")
        require(semver(actual["implementation_version"], "target version") >=
                semver(minimum_version, "minimum version"),
                f"target version is too old for {identity[0]}/{identity[1]}")


def package_model_source_bytes(package: Path) -> int:
    if package.is_dir():
        data, paths = load_recipe_dir(package)
        return sum((package / path).stat().st_size for path in paths)
    content = package.read_bytes()[:HEADER_SIZE]
    _, _, _, _, _, manifest_size, count, package_size = HEADER.unpack_from(content)
    return package_size - HEADER_SIZE - count * ENTRY_SIZE - manifest_size


def fusion_groups(data: dict[str, Any]) -> list[tuple[int, int, int]]:
    pipeline = data["pipeline"]
    groups: list[tuple[int, int, int]] = []
    index = 0
    while index < len(pipeline):
        execution = pipeline[index].get("execution")
        if not isinstance(execution, dict) or execution.get("mode") != "per_item":
            index += 1
            continue
        maximum = execution["max_items"]
        end = index
        while end + 1 < len(pipeline):
            next_execution = pipeline[end + 1].get("execution")
            if (not isinstance(next_execution, dict) or
                    next_execution.get("mode") != "per_item" or
                    next_execution.get("max_items") != maximum):
                break
            end += 1
        groups.append((index, end, maximum))
        index = end + 1
    return groups


def print_package_info(package: Path, data: dict[str, Any],
                       target: Path | None) -> None:
    validate_target(data, target)
    print(f"path: {package}")
    print(f"container: {'recipe directory' if package.is_dir() else 'v3'}")
    print(f"schema: {data['schema']}")
    print(f"runtime_min_version: {data.get('requires', {}).get('runtime_min_version', RUNTIME_VERSION)}")
    print(f"backends: {', '.join(sorted({m['backend'] for m in data['models']}))}")
    print(f"name: {data['name']}")
    print(f"input: {data['input']['format']}")
    print(f"result_capacity: {data['task']['result_capacity']}")
    print(f"label_sets: {', '.join(data['label_sets'])}")
    print(f"model_source_bytes: {package_model_source_bytes(package)}")
    print("model_io_bytes: runtime-only (query mipk_pipeline_get_memory_plan after open)")
    print("pipeline:")
    for index, stage in enumerate(data["pipeline"]):
        execution = stage.get("execution")
        mode = "once"
        if isinstance(execution, dict) and execution.get("mode") == "per_item":
            mode = f"per_item max_items={execution['max_items']}"
        node_name = stage.get(stage["type"], "")
        print(f"  [{index}] {stage['id']}: {stage['type']}/{node_name} [{mode}]")
    groups = fusion_groups(data)
    print(f"fusion_groups: {len(groups)}")
    for number, (start, end, maximum) in enumerate(groups):
        first = data["pipeline"][start]["id"]
        last = data["pipeline"][end]["id"]
        print(f"  [{number}] {first} -> {last}, max_items={maximum}")
    collection_outputs: list[tuple[str, int]] = []
    group_ends = {end: maximum for _, end, maximum in groups}
    for index, stage in enumerate(data["pipeline"]):
        node = capability(stage["type"], stage.get(stage["type"], "")) \
            if stage["type"] != "model" else None
        if node is not None and node.output == "collection":
            collection_outputs.append(
                (stage["output"], stage[node.collection_bound_option or "max_items"]))
        if index in group_ends:
            collection_outputs.append((stage["output"], group_ends[index]))
    print(f"collection_item_slots: {sum(item[1] for item in collection_outputs)}")
    for output, maximum in collection_outputs:
        print(f"  {output}: max_items={maximum}")
    print("memory_budget: static package/collection bounds only; exact CPU/MMZ plan is runtime-only")
    print("target_capabilities: verified" if target else "target_capabilities: not verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-dir", type=Path, action="append", default=[],
                        help="extra directory holding <module>/module.json")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="pack a manifest and its models into a .mipk")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--model", type=Path)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--target-capabilities", type=Path)
    create.add_argument("--no-pin", action="store_true",
                        help="do not fill requires.capabilities from the modules used")
    create.add_argument("--backend",
                        help="override models[].backend, e.g. svp_acl with --model model.om")
    validate = commands.add_parser("validate", help="check a .mipk or recipe directory")
    validate.add_argument("package", type=Path)
    validate.add_argument("--target-capabilities", type=Path)
    inspect = commands.add_parser("inspect", help="describe a .mipk or recipe directory")
    inspect.add_argument("package", type=Path)
    inspect.add_argument("--target-capabilities", type=Path)
    commands.add_parser("recipes", help="list the standard pipelines and their parameters")
    commands.add_parser("modules", help="list known modules and their capabilities")
    new = commands.add_parser("new", help="write a manifest from a model family's standard pipeline")
    new.add_argument("recipe", help="recipe name, e.g. yolo")
    new.add_argument("--output", type=Path, required=True, help="directory to create")
    new.add_argument("--model", help="model file (recipes with one model)")
    new.add_argument("--labels", help="label file or comma-separated list")
    new.add_argument("--backend", help="backend module, default onnxruntime")
    new.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                     help="any recipe parameter; see `recipes`")
    new.add_argument("--link", action="store_true", help="symlink model files instead of copying")
    render = commands.add_parser("render", help="rewrite manifest.json from a directory's recipe.json")
    render.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.module_dir:
        CAPABILITIES.update(load_modules(args.module_dir))
        RECIPES.update(load_recipes(args.module_dir))
    try:
        if args.command == "create":
            data, _ = load_manifest(args.manifest)
            validate_target(data, args.target_capabilities)
            create_package(args.manifest, args.model, args.output, not args.no_pin,
                           args.backend)
            print(f"created: {args.output}")
        elif args.command in {"validate", "inspect"}:
            data = read_package(args.package)
            validate_target(data, args.target_capabilities)
            if args.command == "validate":
                print(f"valid: {args.package}")
            else:
                print_package_info(args.package, data, args.target_capabilities)
        elif args.command == "recipes":
            for name, recipe in sorted(RECIPES.items()):
                print(f"{name}: {recipe.description}")
                for param, spec in recipe.params.items():
                    default = spec.get("default", "(derived)" if "derive" in spec else
                                       "(required)" if spec.get("required") else "-")
                    print(f"    {param:<20} {spec.get('type', 'json'):<8} default={default}"
                          f"  {spec.get('help', '')}")
        elif args.command == "new":
            params = parse_set(args.set)
            if args.model is not None:
                params["model"] = args.model
            if args.labels is not None:
                params["labels"] = args.labels
            if args.backend is not None:
                params["backend"] = args.backend
            new_recipe(args.recipe, args.output, params, args.link)
            print(f"created: {args.output}/manifest.json")
        elif args.command == "render":
            render_dir(args.directory)
            print(f"rendered: {args.directory}/manifest.json")
        elif args.command == "modules":
            for (kind, name), node in sorted(CAPABILITIES.items()):
                print(f"{kind}/{name} v{node.implementation_version} abi {node.abi_version}"
                      f" [{node.module}] options: {', '.join(sorted(node.options)) or '-'}")
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"mipk_tool: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
