"""Data loading and loss nodes.

MNIST, CIFAR10, CSV, ImageFolder, TensorFile loaders + MSE/CE losses.
Requires torch at import time.
"""
from __future__ import annotations

import torch

from ...backend.converter import from_torch, to_torch
from ...backend.device import resolve_device
from ...runtime.registry import register
from ..base import BaseNode, NodeInput, NodeOutput, Parameter


# ----------------------------------------------------------------- MNIST
@register("mnist_loader")
class MnistLoaderNode(BaseNode):
    type_name = "mnist_loader"
    label = "MNIST Loader"
    category = "Data"
    inputs = []
    outputs = [
        NodeOutput("images", data_kind="tensor"),
        NodeOutput("labels", data_kind="tensor"),
    ]
    parameters = [
        Parameter("batch_size", kind="scalar", default=32, dtype="int"),
        Parameter("train", kind="scalar", default=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        bs = int(params["batch_size"])
        train = bool(params["train"])
        try:
            import torchvision
            import torchvision.transforms as transforms
            ds = torchvision.datasets.MNIST(root="./data", train=train, download=True, transform=transforms.ToTensor())
            loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train)
            images, labels = next(iter(loader))
            images, labels = images.to(dev), labels.to(dev)
        except Exception:
            images = torch.rand(bs, 1, 28, 28, device=dev)
            labels = torch.randint(0, 10, (bs,), device=dev)
        return {"images": from_torch(images, metadata={"backend": "torch"}), "labels": from_torch(labels, metadata={"backend": "torch"})}


# --------------------------------------------------------------- CIFAR10
@register("cifar10_loader")
class Cifar10LoaderNode(BaseNode):
    type_name = "cifar10_loader"
    label = "CIFAR10 Loader"
    category = "Data"
    inputs = []
    outputs = [
        NodeOutput("images", data_kind="tensor"),
        NodeOutput("labels", data_kind="tensor"),
    ]
    parameters = [
        Parameter("batch_size", kind="scalar", default=32, dtype="int"),
        Parameter("train", kind="scalar", default=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        bs = int(params["batch_size"])
        train = bool(params["train"])
        try:
            import torchvision
            import torchvision.transforms as transforms
            ds = torchvision.datasets.CIFAR10(root="./data", train=train, download=True, transform=transforms.ToTensor())
            loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=train)
            images, labels = next(iter(loader))
            images, labels = images.to(dev), labels.to(dev)
        except Exception:
            images = torch.rand(bs, 3, 32, 32, device=dev)
            labels = torch.randint(0, 10, (bs,), device=dev)
        return {"images": from_torch(images, metadata={"backend": "torch"}), "labels": from_torch(labels, metadata={"backend": "torch"})}


# ------------------------------------------------------------- CSV Loader
@register("csv_loader")
class CsvLoaderNode(BaseNode):
    """Load a CSV file as a tensor. Uses stdlib csv — no pandas dependency."""

    type_name = "csv_loader"
    label = "CSV Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("data", data_kind="tensor")]
    parameters = [
        Parameter("path", kind="scalar", required=True),
        Parameter("delimiter", kind="scalar", default=","),
        Parameter("skip_header", kind="scalar", default=1, dtype="int"),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        import csv
        from pathlib import Path

        dev = resolve_device(params["device"])
        path = Path(params["path"])
        if not path.exists():
            raise ValueError(f"CSV 文件不存在: {path}")
        delim = params.get("delimiter", ",")
        skip = int(params.get("skip_header", 1))
        rows = []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=delim)
            for _ in range(skip):
                next(reader, None)
            for row in reader:
                rows.append([float(v) for v in row])
        if not rows:
            raise ValueError(f"CSV 无数据: {path}")
        data = torch.tensor(rows, dtype=torch.float32, device=dev)
        return {"data": from_torch(data, metadata={"backend": "torch", "source": str(path)})}


# ------------------------------------------------------- Image Folder
@register("image_folder_loader")
class ImageFolderLoaderNode(BaseNode):
    """Load images from a folder. Uses PIL if available; falls back to random."""

    type_name = "image_folder_loader"
    label = "Image Folder Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("images", data_kind="tensor")]
    parameters = [
        Parameter("path", kind="scalar", required=True),
        Parameter("batch_size", kind="scalar", default=8, dtype="int"),
        Parameter("height", kind="scalar", default=224, dtype="int"),
        Parameter("width", kind="scalar", default=224, dtype="int"),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        from pathlib import Path

        dev = resolve_device(params["device"])
        bs = int(params["batch_size"])
        h = int(params["height"])
        w = int(params["width"])
        folder = Path(params["path"])
        try:
            import torchvision
            import torchvision.transforms as transforms

            transform = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.ToTensor(),
            ])
            ds = torchvision.datasets.ImageFolder(str(folder), transform=transform)
            loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
            images, _ = next(iter(loader))
            images = images.to(dev)
        except Exception:
            images = torch.rand(bs, 3, h, w, device=dev)
        return {"images": from_torch(images, metadata={"backend": "torch"})}


# ------------------------------------------------------- Tensor File
@register("tensor_file_loader")
class TensorFileLoaderNode(BaseNode):
    """Load a .pt / .pth tensor file via torch.load."""

    type_name = "tensor_file_loader"
    label = "Tensor File Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("data", data_kind="tensor")]
    parameters = [
        Parameter("path", kind="scalar", required=True),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        from pathlib import Path

        dev = resolve_device(params["device"])
        path = Path(params["path"])
        if not path.exists():
            raise ValueError(f"Tensor 文件不存在: {path}")
        obj = torch.load(path, map_location=str(dev), weights_only=True)
        if not isinstance(obj, torch.Tensor):
            if isinstance(obj, dict):
                # Try first tensor value in a state_dict
                obj = next(v for v in obj.values() if isinstance(v, torch.Tensor))
            elif isinstance(obj, (list, tuple)):
                obj = obj[0]
            else:
                raise ValueError(f"文件内容不是 tensor: {type(obj)}")
        return {"data": from_torch(obj, metadata={"backend": "torch", "source": str(path)})}


# ------------------------------------------------------------- Data Loader
@register("dataloader")
class DataloaderNode(BaseNode):
    type_name = "dataloader"
    label = "Data Loader"
    category = "Data"
    inputs = []
    outputs = [NodeOutput("data", data_kind="tensor")]
    parameters = [
        Parameter("batch_size", kind="scalar", default=32, dtype="int"),
        Parameter("channels", kind="scalar", default=1, dtype="int"),
        Parameter("height", kind="scalar", default=28, dtype="int"),
        Parameter("width", kind="scalar", default=28, dtype="int"),
        Parameter("device", kind="scalar", default="cpu"),
    ]

    def execute(self, inputs, params, context):
        dev = resolve_device(params["device"])
        bs = int(params["batch_size"])
        c = int(params["channels"])
        h = int(params["height"])
        w = int(params["width"])
        data = torch.rand(bs, c, h, w, device=dev)
        return {"data": from_torch(data, metadata={"backend": "torch"})}


# ------------------------------------------------------------- Loss nodes
@register("mse_loss")
class MseLossNode(BaseNode):
    type_name = "mse_loss"
    label = "MSE Loss"
    category = "Loss"
    inputs = [NodeInput("pred", required=True), NodeInput("target", required=True)]
    outputs = [NodeOutput("loss")]
    parameters = []

    def execute(self, inputs, params, context):
        p = to_torch(inputs["pred"])
        t = to_torch(inputs["target"])
        loss = torch.nn.functional.mse_loss(p, t)
        return {"loss": from_torch(loss, metadata={"backend": "torch"})}


@register("cross_entropy_loss")
class CrossEntropyLossNode(BaseNode):
    type_name = "cross_entropy_loss"
    label = "CrossEntropy Loss"
    category = "Loss"
    inputs = [NodeInput("pred", required=True), NodeInput("target", required=True)]
    outputs = [NodeOutput("loss")]
    parameters = []

    def execute(self, inputs, params, context):
        p = to_torch(inputs["pred"])
        t = to_torch(inputs["target"]).long()
        loss = torch.nn.functional.cross_entropy(p, t)
        return {"loss": from_torch(loss, metadata={"backend": "torch"})}
