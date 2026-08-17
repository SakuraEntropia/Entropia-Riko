"""Input (dataset) + preview nodes: image / text / JSON.

- Loaders read a file into a TensorValue (`image_tensor` / `text` / `json`).
- Preview nodes mark / convert a value for display (text, JSON, or an image
  thumbnail carried as base64 PNG in metadata).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ...backend.converter import to_torch
from ...core.tensor import TensorValue
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


def _load_pil_image(path: str, height: int, width: int):
    """Load an image file as a (H, W, 3) float32 tensor in [0, 1] via PIL."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if height and width:
            img = img.resize((width, height))
        arr = torch.tensor(list(img.getdata()), dtype=torch.float32)
        h, w = img.size[1], img.size[0]
        return arr.view(h, w, 3) / 255.0
    except Exception:
        return torch.rand(height or 64, width or 64, 3)


def _image_thumbnail(tensor, max_size: int = 256) -> str | None:
    """Return a base64 PNG data-URL thumbnail of an image tensor, or None."""
    try:
        import base64
        import io

        from PIL import Image
        t = tensor.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
            if t.shape[-1] == 1:
                t = t.squeeze(-1)
            arr = (t.clamp(0, 1) * 255).byte().numpy()
            img = Image.fromarray(arr, mode="L" if arr.ndim == 2 else "RGB")
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ loaders
@register("text_loader")
class TextLoaderNode(BaseNode):
    type_name = "text_loader"
    label = "Text Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("text", data_kind="text")]
    parameters = [Parameter("path", kind="scalar", required=True)]

    def execute(self, inputs, params, context):
        path = Path(params["path"])
        if not path.exists():
            raise ValueError(f"文本文件不存在: {path}")
        return {"text": TensorValue(path.read_text(encoding="utf-8"), kind="text",
                                    metadata={"source": str(path)})}


@register("json_loader")
class JsonLoaderNode(BaseNode):
    type_name = "json_loader"
    label = "JSON Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("data", data_kind="json")]
    parameters = [Parameter("path", kind="scalar", required=True)]

    def execute(self, inputs, params, context):
        path = Path(params["path"])
        if not path.exists():
            raise ValueError(f"JSON 文件不存在: {path}")
        return {"data": TensorValue(json.loads(path.read_text(encoding="utf-8")), kind="json",
                                    metadata={"source": str(path)})}


@register("image_loader")
class ImageLoaderNode(BaseNode):
    type_name = "image_loader"
    label = "Image Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("image", data_kind="image_tensor")]
    parameters = [Parameter("path", kind="scalar", required=True),
                  Parameter("height", default=0, dtype="int"),
                  Parameter("width", default=0, dtype="int")]

    def execute(self, inputs, params, context):
        path = str(params["path"])
        img = _load_pil_image(path, int(params.get("height", 0)), int(params.get("width", 0)))
        return {"image": TensorValue(img.tolist(), shape=tuple(img.shape), kind="image_tensor",
                                     metadata={"source": path})}


# ------------------------------------------------------------------ previews
@register("text_preview")
class TextPreviewNode(BaseNode):
    type_name = "text_preview"
    label = "Text Preview"
    category = "Utility"
    inputs = [NodeInput("x", data_kind="tensor", required=False, default=None)]
    outputs = [NodeOutput("text", data_kind="text")]
    parameters = []

    def execute(self, inputs, params, context):
        v = inputs.get("x")
        if v is None:
            return {"text": TensorValue("(empty)", kind="text")}
        if isinstance(v, TensorValue) and v.data_kind == "text":
            return {"text": v}
        data = v.data if isinstance(v, TensorValue) else v
        return {"text": TensorValue(str(data), kind="text")}


@register("json_preview")
class JsonPreviewNode(BaseNode):
    type_name = "json_preview"
    label = "JSON Preview"
    category = "Utility"
    inputs = [NodeInput("x", data_kind="tensor", required=False, default=None)]
    outputs = [NodeOutput("data", data_kind="json")]
    parameters = [Parameter("indent", default=2, dtype="int")]

    def execute(self, inputs, params, context):
        v = inputs.get("x")
        if v is None:
            return {"data": TensorValue({}, kind="json")}
        data = v.data if isinstance(v, TensorValue) else v
        text = json.dumps(data, ensure_ascii=False, indent=int(params.get("indent", 2)))
        return {"data": TensorValue(text, kind="json")}


@register("image_preview")
class ImagePreviewNode(BaseNode):
    type_name = "image_preview"
    label = "Image Preview"
    category = "Utility"
    inputs = [NodeInput("x", data_kind="tensor", required=True)]
    outputs = [NodeOutput("image", data_kind="image_tensor")]
    parameters = [Parameter("max_size", default=256, dtype="int")]

    def execute(self, inputs, params, context):
        v = inputs["x"]
        t = to_torch(v) if isinstance(v, TensorValue) else v
        thumb = _image_thumbnail(t, int(params.get("max_size", 256)))
        meta = {"backend": "torch"}
        if thumb:
            meta["preview"] = {"image": thumb}
        return {"image": TensorValue(
            t.detach().cpu().tolist(), shape=tuple(t.shape), dtype="float32",
            kind="image_tensor", metadata=meta)}
