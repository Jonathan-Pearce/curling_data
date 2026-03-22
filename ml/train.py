"""Training loop, evaluation, and CLI entry point for the Siamese GNN.

Usage
-----
::

    python -m ml.train                        # train with defaults
    python -m ml.train --epochs 50 --lr 5e-4  # override hyper-parameters
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PipelineConfig
from .data import (
    CurlingShotDataset,
    collate_curling_batch,
    load_and_preprocess,
    split_by_year,
)
from .loss import MultiTaskLoss
from .model import SiameseGNN


# ── helpers ──────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _build_loaders(
    cfg: PipelineConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load data, split, wrap in DataLoaders."""
    print("Loading and preprocessing data …")
    df = load_and_preprocess(cfg)
    train_df, val_df, test_df = split_by_year(df, cfg)
    print(
        f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,} shots"
    )

    train_ds = CurlingShotDataset(train_df)
    val_ds = CurlingShotDataset(val_df)
    test_ds = CurlingShotDataset(test_df)

    loader_kw = dict(collate_fn=collate_curling_batch, drop_last=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, **loader_kw
    )
    return train_loader, val_loader, test_loader


# ── single epoch ─────────────────────────────────────────────────────────────

def _run_epoch(
    model: SiameseGNN,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, float]:
    """Run one epoch (train or eval).  Returns averaged metrics."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_end = 0.0
    total_acc = 0.0
    total_cls = 0.0
    n_batches = 0
    correct_type = 0
    total_type = 0

    ctx = torch.no_grad() if not is_train else _nullctx()
    with ctx:
        for batch in loader:
            # Move to device
            prev_nodes = batch["prev_nodes"].to(device)
            prev_mask = batch["prev_mask"].to(device)
            cur_nodes = batch["cur_nodes"].to(device)
            cur_mask = batch["cur_mask"].to(device)
            metadata = batch["metadata"].to(device)
            is_first = batch["is_first_shot"].to(device)

            end_t1 = batch["end_score_team1"].to(device)
            end_t2 = batch["end_score_team2"].to(device)
            target_end = torch.stack([end_t1, end_t2], dim=-1)
            target_acc = batch["accuracy"].to(device)
            target_type = batch["shot_type_label"].to(device)

            # Masks for valid labels
            acc_mask = (target_acc >= 0).float()
            type_mask = (target_type >= 0).float()

            preds = model(
                prev_nodes, prev_mask, cur_nodes, cur_mask, metadata, is_first
            )

            losses = criterion(
                preds["end_score"],
                target_end,
                preds["accuracy"],
                target_acc,
                acc_mask,
                preds["shot_type"],
                target_type,
                type_mask,
            )

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()

            total_loss += losses["total"].item()
            total_end += losses["end_score"].item()
            total_acc += losses["accuracy"].item()
            total_cls += losses["shot_type"].item()
            n_batches += 1

            # Classification accuracy
            if type_mask.sum() > 0:
                pred_cls = preds["shot_type"][type_mask.bool()].argmax(dim=-1)
                correct_type += (pred_cls == target_type[type_mask.bool()]).sum().item()
                total_type += type_mask.sum().item()

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "end_mse": total_end / max(n_batches, 1),
        "acc_mse": total_acc / max(n_batches, 1),
        "type_ce": total_cls / max(n_batches, 1),
        "type_accuracy": correct_type / max(total_type, 1),
    }
    return metrics


class _nullctx:
    """Trivial context manager that does nothing (for train mode)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ── training loop ────────────────────────────────────────────────────────────

def train(cfg: PipelineConfig) -> Dict[str, float]:
    """Full training run.  Returns test-set metrics."""
    _set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = _build_loaders(cfg)

    model = SiameseGNN(cfg).to(device)
    criterion = MultiTaskLoss().to(device)

    # Optimise both model params and loss log-variances
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: Optional[dict] = None

    print(f"\nTraining for up to {cfg.epochs} epochs (patience={cfg.patience})")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_m = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_m = _run_epoch(model, val_loader, criterion, None, device)
        elapsed = time.time() - t0

        print(
            f"  epoch {epoch:3d} | "
            f"train_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f} | "
            f"end_mse={val_m['end_mse']:.3f}  acc_mse={val_m['acc_mse']:.1f}  "
            f"type_acc={val_m['type_accuracy']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            patience_counter = 0
            best_state = {
                "model": model.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            }
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        criterion.load_state_dict(best_state["criterion"])

        # Save checkpoint
        save_dir = Path(cfg.model_save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, cfg.model_save_path)
        print(f"\nBest model saved → {cfg.model_save_path}  (epoch {best_state['epoch']})")

    # ── Evaluate on test set ────────────────────────────────────────────
    print("\nTest-set evaluation:")
    test_m = _run_epoch(model, test_loader, criterion, None, device)
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")

    return test_m


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Siamese GNN curling model"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.hidden is not None:
        cfg.gnn_hidden_dim = args.hidden
        cfg.head_hidden_dim = args.hidden
    if args.layers is not None:
        cfg.gnn_num_layers = args.layers
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
