"""Keypoint R-CNN model that returns every gate instance in an image."""

from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from src.models.training.targets import KEYPOINT_COUNT


class GateDetector(nn.Module):
    """One-class detector with four side-invariant keypoints per gate."""

    def __init__(
        self,
        pretrained: bool = True,
        input_size: tuple[int, int] = (360, 640),
        score_threshold: float = 0.7,
        detections_per_image: int = 12,
        box_nms_threshold: float = 0.5,
        allow_random_init_on_pretrained_failure: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        anchor_generator = AnchorGenerator(
            sizes=((8,), (16,), (32,), (64,), (128,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5,
        )
        try:
            detector = keypointrcnn_resnet50_fpn(
                weights=weights,
                weights_backbone=None,
                rpn_anchor_generator=anchor_generator,
                min_size=input_size[0],
                max_size=input_size[1],
                box_score_thresh=score_threshold,
                box_nms_thresh=box_nms_threshold,
                box_detections_per_img=detections_per_image,
            )
            self.pretrained_loaded = pretrained
        except (OSError, RuntimeError) as exc:
            if pretrained and not allow_random_init_on_pretrained_failure:
                raise RuntimeError(
                    "Could not load pretrained Keypoint R-CNN weights. "
                    "Check network access or pre-populate the torch cache. "
                    "Use --allow-random-init-on-pretrained-failure only when "
                    "you intentionally want to train from scratch."
                ) from exc
            warnings.warn(
                f"Could not load pretrained Keypoint R-CNN weights ({exc}); "
                "continuing with random initialization.",
                stacklevel=2,
            )
            detector = keypointrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
                num_classes=2,
                num_keypoints=KEYPOINT_COUNT,
                rpn_anchor_generator=anchor_generator,
                min_size=input_size[0],
                max_size=input_size[1],
                box_score_thresh=score_threshold,
                box_nms_thresh=box_nms_threshold,
                box_detections_per_img=detections_per_image,
            )
            self.pretrained_loaded = False

        box_features = detector.roi_heads.box_predictor.cls_score.in_features
        detector.roi_heads.box_predictor = FastRCNNPredictor(
            box_features,
            2,
        )
        keypoint_features = (
            detector.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        )
        detector.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            keypoint_features,
            KEYPOINT_COUNT,
        )

        self.detector = detector
        self.input_size = input_size
        self.normalization = "frozen_batch_norm"

    @property
    def features(self) -> nn.Module:
        """Backbone compatibility for optimizer setup."""

        return self.detector.backbone

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        return self.detector(images, targets)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.detector.backbone.parameters():
            parameter.requires_grad = trainable

    def set_score_threshold(self, threshold: float) -> None:
        self.detector.roi_heads.score_thresh = threshold

    def set_box_nms_threshold(self, threshold: float) -> None:
        self.detector.roi_heads.nms_thresh = threshold


# Existing imports can continue to use the old class name.
GatePoseNet = GateDetector
