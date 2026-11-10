# yolo11n recipe

`recipe.json` is the source: four lines naming the `yolo` standard pipeline,
the model file and the label file. `manifest.json` is rendered from it by
`mipk_tool.py render` (rerun after editing `recipe.json`); edit the manifest
directly only for a custom graph, and delete `recipe.json` when you do.

The model file is not in the repository; produce it with `./export.sh`
(requires `ultralytics`) or export it yourself with `imgsz=640` and static
shapes.

```sh
build/examples/host/mipk_run examples/host/models/yolo11n photo.jpg --out out.png
```
