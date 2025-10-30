from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import torchvision.models._utils as _utils
from torchvision import models

from models.face_detection.retinaface.backbone.mobilenetv2 import mobilenet_v2
from models.face_detection.retinaface.common import (
    FPN,
    SSH,
    IntermediateLayerGetterByIndex,
)


def get_layer_extractor(cfg, backbone):
    if cfg["name"].startswith("mobilenet_v2"):
        return IntermediateLayerGetterByIndex(backbone, [6, 13, 18])
    else:
        raise ValueError(f"Unsupported backbone type: {cfg['name']}")


def build_backbone(name, pretrained=False):
    name = name.lower()
    width_mult_map = {
        "mobilenet_v2_0.25": 0.25,
        "mobilenet_v2_0.5": 0.5,
        "mobilenet_v2_0.75": 0.75,
        "mobilenet_v2_1.0": 1.0,
    }

    if name not in width_mult_map:
        raise ValueError(f"Unsupported backbone name: {name}")

    width_mult = width_mult_map[name]
    backbone = mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
    return backbone


class ClassHead(nn.Module):
    def __init__(
        self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3
    ) -> None:
        super().__init__()
        self.class_head = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_anchors * 2,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                for _ in range(fpn_num)
            ]
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.class_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 2) for out in outputs], dim=1)
        return outputs


class BboxHead(nn.Module):
    def __init__(
        self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3
    ) -> None:
        super().__init__()
        self.bbox_head = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_anchors * 4,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                for _ in range(fpn_num)
            ]
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.bbox_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 4) for out in outputs], dim=1)
        return outputs


class LandmarkHead(nn.Module):
    def __init__(
        self, in_channels: int = 512, num_anchors: int = 2, fpn_num: int = 3
    ) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels,
                    num_anchors * 10,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                )
                for _ in range(fpn_num)
            ]
        )

    def forward(self, x: List[Tensor]) -> Tensor:
        outputs = []
        for feature, layer in zip(x, self.landmark_head):
            outputs.append(layer(feature).permute(0, 2, 3, 1).contiguous())

        outputs = torch.cat([out.view(out.shape[0], -1, 10) for out in outputs], dim=1)
        return outputs


class RetinaFace(nn.Module):
    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided and cannot be None")
        backbone = build_backbone(cfg["name"], cfg["pretrain"])
        self.fx = get_layer_extractor(cfg, backbone)

        num_anchors = 2
        out_channels = cfg["out_channel"]

        fpn_channels_map = {
            "mobilenet_v2_0.25": [8, 24, 320],
            "mobilenet_v2_0.5": [16, 48, 640],
            "mobilenet_v2_0.75": [24, 72, 960],
            "mobilenet_v2_1.0": [32, 96, 1280],
        }

        if cfg["name"] not in fpn_channels_map:
            raise ValueError(f"Unsupported backbone: {cfg['name']}")

        fpn_in_channels = fpn_channels_map[cfg["name"]]

        self.fpn = FPN(fpn_in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.class_head = ClassHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )
        self.bbox_head = BboxHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )
        self.landmark_head = LandmarkHead(
            in_channels=cfg["out_channel"], num_anchors=num_anchors, fpn_num=3
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out = self.fx(x)
        fpn = self.fpn(out)

        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])

        features = [feature1, feature2, feature3]

        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)

        if self.training:
            output = (bbox_regressions, classifications, landmark_regressions)
        else:
            output = (
                bbox_regressions,
                F.softmax(classifications, dim=-1),
                landmark_regressions,
            )
        return output
