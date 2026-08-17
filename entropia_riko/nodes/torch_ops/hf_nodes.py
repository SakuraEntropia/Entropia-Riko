"""Hugging Face / pretrained model nodes (optional backends).

- ``diffusers_text2img`` — image generation via the Diffusers library.
- ``transformers_pipeline`` — any Hugging Face ``pipeline`` (text generation,
  classification, fill-mask, …).
- ``transformers_embedding`` — a pretrained encoder (including JEPA-class /
  self-supervised models) → token embeddings.

These libraries are **optional**: ``import diffusers`` / ``import transformers``
is deferred to ``execute()``, and failures (missing package / no network) fall
back to a placeholder with a ``warning`` in metadata so graphs still run.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import torch

from ..base import BaseNode, NodeInput, NodeOutput, Parameter
from ...runtime.registry import register
from ...core.tensor import TensorValue
from ...backend.converter import from_torch


def _warn(msg: str) -> Dict[str, Any]:
    return {"backend": "huggingface", "warning": str(msg)[:200]}


@register("diffusers_text2img")
class DiffusersText2ImgNode(BaseNode):
    type_name = "diffusers_text2img"
    label = "Diffusers Text→Image"
    category = "HuggingFace"
    inputs = []
    outputs = [NodeOutput("image", data_kind="image_tensor")]
    parameters = [
        Parameter("model_id", kind="scalar", required=True, default="stabilityai/sdxl-turbo"),
        Parameter("prompt", kind="scalar", required=True, default="a red fox"),
        Parameter("num_steps", kind="scalar", default=4, dtype="int"),
        Parameter("guidance_scale", kind="scalar", default=0.0, dtype="float"),
        Parameter("height", kind="scalar", default=512, dtype="int"),
        Parameter("width", kind="scalar", default=512, dtype="int"),
    ]

    def execute(self, inputs, params, context):
        h = int(params.get("height", 512))
        w = int(params.get("width", 512))
        try:
            from diffusers import DiffusionPipeline

            pipe = DiffusionPipeline.from_pretrained(params["model_id"])
            img = pipe(
                params["prompt"],
                num_inference_steps=int(params.get("num_steps", 4)),
                guidance_scale=float(params.get("guidance_scale", 0.0)),
                height=h,
                width=w,
            ).images[0].convert("RGB")
            arr = torch.tensor(list(img.getdata()), dtype=torch.float32).view(img.size[1], img.size[0], 3) / 255.0
            return {"image": TensorValue(arr.tolist(), shape=tuple(arr.shape), kind="image_tensor",
                                         metadata={"backend": "diffusers", "model": params["model_id"]})}
        except Exception as exc:  # noqa: BLE001 - missing lib / network → placeholder
            return {"image": TensorValue(torch.rand(h, w, 3).tolist(), shape=(h, w, 3), kind="image_tensor",
                                         metadata=_warn(f"diffusers 生成失败: {exc}"))}


@register("transformers_pipeline")
class TransformersPipelineNode(BaseNode):
    type_name = "transformers_pipeline"
    label = "Transformers Pipeline"
    category = "HuggingFace"
    inputs = []
    outputs = [NodeOutput("result", data_kind="text")]
    parameters = [
        Parameter("task", kind="scalar", required=True, default="sentiment-analysis",
                  choices=["sentiment-analysis", "text-classification", "text-generation", "fill-mask", "feature-extraction"]),
        Parameter("model_id", kind="scalar", required=True, default="distilbert-base-uncased-finetuned-sst-2-english"),
        Parameter("text", kind="scalar", required=True, default="I love this."),
    ]

    def execute(self, inputs, params, context):
        try:
            from transformers import pipeline

            p = pipeline(params["task"], model=params["model_id"])
            out = p(params["text"])
            text = json.dumps(out, ensure_ascii=False, indent=2)
            return {"result": TensorValue(text, kind="text",
                                          metadata={"backend": "transformers", "model": params["model_id"]})}
        except Exception as exc:  # noqa: BLE001
            return {"result": TensorValue(f"(transformers 不可用或加载失败) {exc}", kind="text",
                                          metadata=_warn(str(exc)))}


@register("transformers_embedding")
class TransformersEmbeddingNode(BaseNode):
    type_name = "transformers_embedding"
    label = "Transformers Embedding (encoder / JEPA)"
    category = "HuggingFace"
    inputs = []
    outputs = [NodeOutput("embedding", data_kind="tensor")]
    parameters = [
        Parameter("model_id", kind="scalar", required=True, default="bert-base-uncased"),
        Parameter("text", kind="scalar", required=True, default="hello world"),
        Parameter("pool", kind="scalar", default="mean", choices=["mean", "cls"]),
    ]

    def execute(self, inputs, params, context):
        try:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(params["model_id"])
            model = AutoModel.from_pretrained(params["model_id"]).eval()
            enc = tok(params["text"], return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = model(**enc)
            if hasattr(out, "last_hidden_state"):
                h = out.last_hidden_state
                h = h.mean(1) if params.get("pool", "mean") == "mean" else h[:, 0]
            elif hasattr(out, "pooler_output"):
                h = out.pooler_output
            else:
                h = torch.zeros(1, 1)
            return {"embedding": from_torch(h[0], metadata={"backend": "transformers", "model": params["model_id"]})}
        except Exception as exc:  # noqa: BLE001
            return {"embedding": TensorValue([0.0], shape=(1,), metadata=_warn(str(exc)))}
