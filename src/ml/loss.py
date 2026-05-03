"""Homoscedastic uncertainty loss for multi-task learning.

Implements the log-likelihood loss weighting from:

    Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics", CVPR 2018.
    https://arxiv.org/abs/1705.07115

For each task *i* with raw per-task loss ``L_i``, the contribution to the
combined loss is:

* **Regression task**:
  ``0.5 * exp(−log_s_i) * L_i + 0.5 * log_s_i``
* **Classification task**:
  ``exp(−log_s_i) * L_i + log_s_i``

where ``log_s_i = log(σ_i²)`` is a *learned* parameter.  This is equivalent
to Equation (3) of Kendall et al. when parameterised as the log-variance.

Importable API::

    from ml.loss import HomoscedasticLoss, compute_loss
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HomoscedasticLoss", "compute_loss"]


class HomoscedasticLoss(nn.Module):
    """Composite multi-task loss with learnable homoscedastic uncertainty.

    Maintains one ``log_s_i = log(σ_i²)`` parameter per task.  All
    parameters are initialised to zero (σ = 1, equal initial weighting).

    Parameters
    ----------
    n_tasks : int
        Number of tasks.  The pipeline uses **3**: shot accuracy (regression),
        end score (regression), and shot type (classification).
    """

    def __init__(self, n_tasks: int = 3) -> None:
        super().__init__()
        # One learnable log-variance per task; shape (n_tasks,)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    @property
    def task_weights(self) -> torch.Tensor:
        """Per-task σ² values (detached; for logging only)."""
        return torch.exp(self.log_vars).detach()

    def forward(
        self,
        loss_acc: torch.Tensor,    # scalar — accuracy MSE
        loss_score: torch.Tensor,  # scalar — end-score MSE
        loss_type: torch.Tensor,   # scalar — shot-type cross-entropy
    ) -> torch.Tensor:
        """Combine three per-task losses with uncertainty weighting.

        All three arguments must be scalar (0-d) tensors.  Any NaN /
        missing-data masking must be applied *before* calling this method.

        Returns
        -------
        torch.Tensor
            Scalar combined loss ready for ``.backward()``.
        """
        s0, s1, s2 = self.log_vars[0], self.log_vars[1], self.log_vars[2]

        # Regression tasks: factor ½ difference vs. classification (Eq. 3)
        l_acc   = 0.5 * torch.exp(-s0) * loss_acc   + 0.5 * s0
        l_score = 0.5 * torch.exp(-s1) * loss_score + 0.5 * s1
        l_type  =       torch.exp(-s2) * loss_type  +       s2

        return l_acc + l_score + l_type


def compute_loss(
    preds: dict[str, torch.Tensor],
    batch_labels: dict[str, torch.Tensor],
    homo_loss: HomoscedasticLoss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the composite multi-task loss for one mini-batch.

    Handles missing labels gracefully:

    * NaN ``accuracy`` values → sample excluded from accuracy loss.
    * ``shot_type_idx == −1`` → sample excluded from shot-type loss.
    * NaN end scores → sample excluded from end-score loss.

    Parameters
    ----------
    preds : dict
        Output of ``CurlingGNN.forward``:
        ``accuracy`` (B, 1), ``shot_type`` (B, N_SHOT_TYPES),
        ``end_scores`` (B, 2).
    batch_labels : dict
        Tensors keyed by ``y_accuracy`` (B,), ``y_shot_type`` (B,),
        ``y_team1_score`` (B,), ``y_team2_score`` (B,).
    homo_loss : HomoscedasticLoss

    Returns
    -------
    total_loss : torch.Tensor
        Scalar combined loss (with gradient).
    metrics : dict[str, float]
        Raw per-task loss values for logging (detached floats).
    """
    device = preds["accuracy"].device

    # --- Shot accuracy (regression, MSE) ---
    y_acc = batch_labels["y_accuracy"]
    acc_mask = ~torch.isnan(y_acc)
    if acc_mask.any():
        acc_mse = F.mse_loss(
            preds["accuracy"].squeeze(-1)[acc_mask], y_acc[acc_mask]
        )
    else:
        acc_mse = torch.zeros(1, device=device, requires_grad=True).squeeze()

    # --- End score (regression, MSE) ---
    y_t1 = batch_labels["y_team1_score"]
    y_t2 = batch_labels["y_team2_score"]
    score_mask = ~(torch.isnan(y_t1) | torch.isnan(y_t2))
    if score_mask.any():
        pred_sc = preds["end_scores"][score_mask]
        true_sc = torch.stack([y_t1[score_mask], y_t2[score_mask]], dim=-1)
        score_mse = F.mse_loss(pred_sc, true_sc)
    else:
        score_mse = torch.zeros(1, device=device, requires_grad=True).squeeze()

    # --- Shot type (classification, cross-entropy) ---
    y_type = batch_labels["y_shot_type"]
    type_mask = y_type >= 0  # −1 signals unknown / missing
    if type_mask.any():
        type_ce = F.cross_entropy(
            preds["shot_type"][type_mask], y_type[type_mask]
        )
    else:
        type_ce = torch.zeros(1, device=device, requires_grad=True).squeeze()

    total_loss = homo_loss(acc_mse, score_mse, type_ce)

    metrics: dict[str, float] = {
        "loss_accuracy": acc_mse.item(),
        "loss_end_score": score_mse.item(),
        "loss_shot_type": type_ce.item(),
    }
    return total_loss, metrics
