"""Homoscedastic-uncertainty multi-task loss (Kendall et al., 2018).

Each task *i* has a learnable log-variance parameter ``log_var_i``.
The composite loss is::

    L = Σ_i  (1 / (2 · exp(log_var_i))) · L_i  +  log_var_i / 2

This automatically balances gradient magnitudes across tasks with
very different loss scales (MSE vs cross-entropy).

Reference
---------
Kendall, A., Gal, Y., & Cipolla, R. (2018).
*Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry
and Semantics.* CVPR 2018.  https://arxiv.org/abs/1705.07115
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """Three-task loss with learnable homoscedastic uncertainty weights.

    Tasks
    -----
    0. End score regression  (MSE)
    1. Shot accuracy regression  (MSE)
    2. Shot type classification  (Cross-entropy)
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialise log-variances to 0  →  initial weight = 1/(2·1) = 0.5
        self.log_var_end = nn.Parameter(torch.zeros(1))
        self.log_var_acc = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        pred_end: torch.Tensor,
        target_end: torch.Tensor,
        pred_acc: torch.Tensor,
        target_acc: torch.Tensor,
        acc_mask: torch.Tensor,
        pred_type: torch.Tensor,
        target_type: torch.Tensor,
        type_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        pred_end : (B, 2)
        target_end : (B, 2)
        pred_acc : (B, 1)
        target_acc : (B,)
        acc_mask : (B,)   1 where accuracy label is valid, 0 otherwise
        pred_type : (B, C)
        target_type : (B,)  long class indices
        type_mask : (B,)   1 where shot-type label is valid, 0 otherwise

        Returns
        -------
        dict with ``total``, ``end_score``, ``accuracy``, ``shot_type``,
        and effective weights ``w_end``, ``w_acc``, ``w_cls``.
        """
        # ── End score MSE (always valid) ────────────────────────────────
        loss_end = F.mse_loss(pred_end, target_end)

        # ── Accuracy MSE (masked) ───────────────────────────────────────
        if acc_mask.sum() > 0:
            pred_acc_m = pred_acc.squeeze(-1)[acc_mask.bool()]
            target_acc_m = target_acc[acc_mask.bool()]
            loss_acc = F.mse_loss(pred_acc_m, target_acc_m)
        else:
            loss_acc = torch.tensor(0.0, device=pred_acc.device)

        # ── Shot type cross-entropy (masked) ────────────────────────────
        if type_mask.sum() > 0:
            pred_type_m = pred_type[type_mask.bool()]
            target_type_m = target_type[type_mask.bool()]
            loss_cls = F.cross_entropy(pred_type_m, target_type_m)
        else:
            loss_cls = torch.tensor(0.0, device=pred_type.device)

        # ── Homoscedastic weighting ─────────────────────────────────────
        precision_end = torch.exp(-self.log_var_end)
        precision_acc = torch.exp(-self.log_var_acc)
        precision_cls = torch.exp(-self.log_var_cls)

        total = (
            precision_end * loss_end + self.log_var_end / 2
            + precision_acc * loss_acc + self.log_var_acc / 2
            + precision_cls * loss_cls + self.log_var_cls / 2
        ).squeeze()

        return {
            "total": total,
            "end_score": loss_end.detach(),
            "accuracy": loss_acc.detach(),
            "shot_type": loss_cls.detach(),
            "w_end": precision_end.detach().squeeze(),
            "w_acc": precision_acc.detach().squeeze(),
            "w_cls": precision_cls.detach().squeeze(),
        }
