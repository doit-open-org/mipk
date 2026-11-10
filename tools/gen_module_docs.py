#!/usr/bin/env python3
"""Generates docs/modules.md from every module.json and recipe.json.

Run after changing a module's options; CI checks the output is current:
    python3 tools/gen_module_docs.py            # write docs/modules.md
    python3 tools/gen_module_docs.py --check    # exit 1 if it is stale
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "modules.md"


def describe_spec(spec: dict) -> str:
    parts = [spec.get("type", "json")]
    if "enum" in spec:
        parts.append("one of " + ", ".join(f"`{item}`" for item in spec["enum"]))
    bounds = []
    if "min" in spec:
        bounds.append(f">= {spec['min']}")
    if "exclusive_min" in spec:
        bounds.append(f"> {spec['exclusive_min']}")
    if "max" in spec:
        bounds.append(f"<= {spec['max']}")
    if bounds:
        parts.append(" ".join(bounds))
    if spec.get("length"):
        parts.append(f"length {spec['length']}")
    if spec.get("items"):
        parts.append(f"items {spec['items']}")
    if spec.get("unique"):
        parts.append("unique")
    return ", ".join(parts)


def render() -> str:
    lines = ["# 模块参考", "",
             "由 `tools/gen_module_docs.py` 从各模块的 `module.json` 与 `recipe.json` 生成，请勿手改。", ""]
    lines += ["## 能力一览", "", "| 种类 | 名称 | 模块 | 版本 | 兼容自 | 输入端口 | 输出 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    capabilities = []
    for base in ("ops", "models", "backends"):
        for path in sorted((ROOT / base).glob("*/module.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for cap in data["capabilities"]:
                capabilities.append((base, data["module"], cap))
    for base, module, cap in capabilities:
        output = cap.get("output", "-")
        if isinstance(cap.get("result_types"), list):
            output += " (" + ", ".join(cap["result_types"]) + ")"
        elif isinstance(cap.get("result_types"), str):
            output += f" ({cap['result_types']})"
        elif cap.get("collection_item"):
            output += f" of {cap['collection_item']}"
        lines.append(f"| {cap['kind']} | `{cap['name']}` | `{base}/{module}` | {cap['implementation_version']} | "
                     f"{cap.get('compat_version', cap['implementation_version'])} | "
                     f"{', '.join(f'`{p}`' for p in cap.get('inputs', [])) or '-'} | {output} |")
    lines.append("")
    lines += ["## 节点选项", ""]
    for base, module, cap in capabilities:
        options = cap.get("options", {})
        if not options:
            continue
        lines += [f"### `{cap['name']}` ({cap['kind']}, `{base}/{module}`)", ""]
        lines += ["| 选项 | 类型与范围 | 必填 | 默认 | 说明 |", "| --- | --- | --- | --- | --- |"]
        for name, spec in options.items():
            default = json.dumps(spec["default"]) if "default" in spec else "-"
            lines.append(f"| `{name}` | {describe_spec(spec)} | {'是' if spec.get('required') else ''} | "
                         f"`{default}` | {spec.get('help', '')} |")
        for constraint in cap.get("constraints", []):
            kind, fields = next(iter(constraint.items()))
            text = {"order": "非递减", "le": "前者不大于后者", "together": "要么都写要么都不写"}[kind]
            lines.append(f"| 约束 | {kind}: {', '.join(f'`{f}`' for f in fields)} | | | {text} |")
        if cap.get("accepts"):
            for port, types in cap["accepts"].items():
                lines.append(f"| 端口 `{port}` | 接受 {', '.join(types)} 结果 | | | |")
        if cap.get("result_bound_option"):
            lines.append(f"| 结果条数 | 由 `{cap['result_bound_option']}` 限制 | | | |")
        lines.append("")
    lines += ["## 标准流水线（recipe）", "",
              "`python3 tools/mipk_tool.py new <recipe> --model <file> --labels <file|a,b> --output <dir> [--set name=value]`", ""]
    for path in sorted((ROOT / "models").glob("*/recipe.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        lines += [f"### `{data['name']}`", "", data.get("description", ""), "",
                  "| 参数 | 类型 | 默认 | 说明 |", "| --- | --- | --- | --- |"]
        for name, spec in data["params"].items():
            if "derive" in spec:
                derive = spec["derive"]
                rule = derive["expr"] if "expr" in derive else f"{derive['fn']}({', '.join(derive['args'])})"
                default = f"由 `{rule}` 推导"
            elif spec.get("required"):
                default = "必填"
            else:
                default = f"`{json.dumps(spec.get('default'))}`"
            lines.append(f"| `{name}` | {spec.get('type', 'json')} | {default} | {spec.get('help', '')} |")
        stages = " → ".join(f"`{stage.get('operator') or stage.get('postprocess') or 'model'}`"
                            + (" (可选)" if "$if" in stage else "")
                            for stage in data["manifest"]["pipeline"])
        lines += ["", f"流水线：{stages}", ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != text:
            print("docs/modules.md is stale; run tools/gen_module_docs.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
