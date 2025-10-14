from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

# Assuming mmrazor.registry and MODELS are part of the framework you're using

global_image_counter = 0

class DCSFKDLoss(nn.Module):
    def __init__(self, loss_weight=1.0, resize_stu=True, align_target: str = "student"):
        super(DCSFKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu
        assert align_target in ("student", "teacher")
        self.align_target = align_target
        self.align_layers = nn.ModuleDict()

    def _get_align(self, in_c: int, out_c: int) -> Union[nn.Conv2d, None]:
        """in_c→out_c 채널 정렬기(1×1 conv)를 가져오거나 새로 만든다."""
        if in_c == out_c:
            return None
        key = f"{in_c}->{out_c}"
        if key not in self.align_layers:
            layer = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
            self.align_layers[key] = layer
        return self.align_layers[key]

    @torch.no_grad()
    def warmup_from_feats(self, feat_S: torch.Tensor, feat_T: torch.Tensor):
        cS, cT = feat_S.shape[1], feat_T.shape[1]
        if self.align_target == "student":
            _ = self._get_align(cT, cS)
        else:
            _ = self._get_align(cS, cT)

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances."""
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S: Union[torch.Tensor, Tuple], preds_T: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        global global_image_counter
        """Forward computation."""
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T = (preds_S, ), (preds_T, )

        loss = 0.0

        for pred_S, pred_T in zip(preds_S, preds_T):
            cS, cT = pred_S.shape[1], pred_T.shape[1]

            if self.align_target == "student":
                align = self._get_align(cT, cS)
                if align is not None:
                    align = align.to(pred_T.device, dtype=pred_T.dtype)
                    pred_T = align(pred_T)
            else:  # align_target == "teacher": student→teacher
                align = self._get_align(cS, cT)
                if align is not None:
                    align = align.to(pred_S.device, dtype=pred_S.dtype)
                    pred_S = align(pred_S)

            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]

            if size_S != size_T:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear', align_corners=False)
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear', align_corners=False)

            assert pred_S.shape == pred_T.shape

            # Apply normalization
            norm_S, norm_T = self.norm(pred_S), self.norm(pred_T)

            with torch.no_grad():
                gap_T = F.adaptive_avg_pool2d(norm_T, (1, 1)).view(norm_T.size(0), -1)
                softmax_att_T = F.softmax(gap_T, dim=-1) * 9 + 1

            att_T = softmax_att_T.view(norm_S.size(0), norm_S.size(1), 1, 1).expand_as(norm_S)

            weighted_S = norm_S * att_T

            # Flatten the feature maps for MSE computation
            weighted_S = weighted_S.view(weighted_S.size(0), weighted_S.size(1), -1)
            norm_T = norm_T.view(norm_T.size(0), norm_T.size(1), -1)

            # Compute MSE loss with teacher attentions applied to student feature maps
            loss += F.mse_loss(weighted_S, norm_T)

        return loss * self.loss_weight
