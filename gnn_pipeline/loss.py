"""Multi-task loss with homoscedastic uncertainty weighting.

Implements the method from Kendall et al. (2018):
    *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics*  (https://arxiv.org/abs/1705.07115)

For each task *i* with raw loss ``L_i``, the weighted contribution is::

    (1 / (2 * σ_i²)) * L_i  +  log(σ_i)

where ``σ_i`` is a learned parameter (stored as ``log_var_i = log(σ_i²)``
for numerical stability).  This lets the optimiser automatically balance
gradients across regression and classification objectives.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """Learnable multi-task loss combining MSE and cross-entropy terms."""

    def __init__(self) -> None:
        super().__init__()
        # Learnable log-variance parameters (initialised to 0 → σ² = 1).
        self.log_var_accuracy = nn.Parameter(torch.zeros(1))
        self.log_var_shot_type = nn.Parameter(torch.zeros(1))
        self.log_var_end_score = nn.Parameter(torch.zeros(1))

        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        accuracy_pred: torch.Tensor,
        accuracy_target: torch.Tensor,
        accuracy_mask: torch.Tensor,
        shot_type_logits: torch.Tensor,
        shot_type_target: torch.Tensor,
        shot_type_mask: torch.Tensor,
        end_score_pred: torch.Tensor,
        end_score_target: torch.Tensor,
        end_score_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite loss.

        Each ``*_mask`` tensor is a boolean ``[B]`` indicating which samples
        in the batch have valid targets for that task (some targets may be
        missing in a given row).

        Returns
        -------
        total_loss : scalar tensor (for ``.backward()``)
        details    : dict of individual un-weighted loss values (floats)
        """
        details: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=accuracy_pred.device)

        # --- Shot accuracy (MSE) -----------------------------------------
        if accuracy_mask.any():
            acc_loss = self.mse(
                accuracy_pred[accuracy_mask], accuracy_target[accuracy_mask]
            )
            precision = torch.exp(-self.log_var_accuracy)
            total_loss = total_loss + precision * acc_loss + self.log_var_accuracy
            details["accuracy_loss"] = acc_loss.item()

        # --- Shot type (cross-entropy) ------------------------------------
        if shot_type_mask.any():
            type_loss = self.ce(
                shot_type_logits[shot_type_mask], shot_type_target[shot_type_mask]
            )
            precision = torch.exp(-self.log_var_shot_type)
            total_loss = total_loss + precision * type_loss + self.log_var_shot_type
            details["shot_type_loss"] = type_loss.item()

        # --- End score (MSE) ----------------------------------------------
        if end_score_mask.any():
            score_loss = self.mse(
                end_score_pred[end_score_mask], end_score_target[end_score_mask]
            )
            precision = torch.exp(-self.log_var_end_score)
            total_loss = total_loss + precision * score_loss + self.log_var_end_score
            details["end_score_loss"] = score_loss.item()

        details["total_loss"] = total_loss.item()
        return total_loss, details
