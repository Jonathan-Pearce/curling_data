"""Training and evaluation loop for the Siamese GNN pipeline.

Run directly::

    python -m gnn_pipeline.train --shots_csv output/shot_locations.csv \\
                                  --ends_csv output/ends.csv
"""

import argparse
import math
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from gnn_pipeline.config import GNNConfig
from gnn_pipeline.data import CurlingDataset, load_and_preprocess
from gnn_pipeline.loss import MultiTaskLoss
from gnn_pipeline.model import SiameseGNN


# ── Collate ──────────────────────────────────────────────────────────────


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack individual samples into batched tensors."""
    return {k: torch.stack([s[k] for s in batch]) for k in batch[0]}


# ── Single epoch helpers ─────────────────────────────────────────────────


def _run_epoch(
    model: SiameseGNN,
    criterion: MultiTaskLoss,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    """Run one epoch (train if *optimizer* is given, else eval)."""
    is_train = optimizer is not None
    model.train(is_train)

    running: Dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        accuracy_pred, shot_type_logits, end_score_pred = model(
            batch["prev_node_feat"],
            batch["prev_adj"],
            batch["prev_mask"],
            batch["prev_is_null"],
            batch["curr_node_feat"],
            batch["curr_adj"],
            batch["curr_mask"],
            batch["metadata"],
        )

        # Build masks for valid targets
        acc_mask = ~torch.isnan(batch["accuracy"])
        type_mask = batch["shot_type"] >= 0
        score_mask = ~torch.isnan(batch["end_score"])

        loss, details = criterion(
            accuracy_pred,
            batch["accuracy"],
            acc_mask,
            shot_type_logits,
            batch["shot_type"],
            type_mask,
            end_score_pred,
            batch["end_score"],
            score_mask,
        )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for k, v in details.items():
            running[k] = running.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


# ── Public API ───────────────────────────────────────────────────────────


def train(
    config: GNNConfig,
    shots_csv: str,
    ends_csv: str,
) -> Tuple[SiameseGNN, Dict[str, List[float]]]:
    """Full training run.  Returns the trained model and loss history."""
    device = torch.device(config.device)

    # Data
    samples = load_and_preprocess(shots_csv, ends_csv, config)
    dataset = CurlingDataset(samples)

    n_total = len(dataset)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = SiameseGNN(config).to(device)
    criterion = MultiTaskLoss().to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = _run_epoch(model, criterion, train_loader, device, optimizer)
        val_metrics = _run_epoch(model, criterion, val_loader, device)

        history["train_loss"].append(train_metrics.get("total_loss", math.nan))
        history["val_loss"].append(val_metrics.get("total_loss", math.nan))

        print(
            f"Epoch {epoch:3d}/{config.num_epochs}  "
            f"train_loss={train_metrics.get('total_loss', 0):.4f}  "
            f"val_loss={val_metrics.get('total_loss', 0):.4f}"
        )

    return model, history


# ── CLI entry-point ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Siamese GNN for curling analysis")
    parser.add_argument("--shots_csv", default="output/shot_locations.csv")
    parser.add_argument("--ends_csv", default="output/ends.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = GNNConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    train(config, args.shots_csv, args.ends_csv)


if __name__ == "__main__":
    main()
