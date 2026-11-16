from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from grain_cnn_model import GrainCNNExperiment, count_parameters, find_best_threshold
from grain_data_loader import build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom CNN for columnar vs. equiaxed grain classification.")
    parser.add_argument("--dataset-root", type=str, default="Dataset_merged")
    parser.add_argument("--split-dir", type=str, default="artifacts/splits")
    parser.add_argument("--output-dir", type=str, default="artifacts/grain_cnn_run")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--reduce-lr-patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-mode", choices=("rgb", "shape"), default="rgb", help="Use RGB or grayscale shape-only input.")
    parser.add_argument("--image-size", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument(
        "--imbalance-correction",
        choices=("sampler", "loss", "none"),
        default="sampler",
        help="Apply class balancing once via a sampler, weighted loss, or not at all.",
    )
    return parser.parse_args()


def save_history(history: list[dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    train_f1 = [row["train_macro_f1"] for row in history]
    val_f1 = [row["val_macro_f1"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, train_loss, label="train loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="val loss", linewidth=2)
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train accuracy", linewidth=2)
    axes[1].plot(epochs, val_acc, label="val accuracy", linewidth=2)
    axes[1].plot(epochs, train_f1, label="train macro F1", linewidth=1.5, linestyle="--")
    axes[1].plot(epochs, val_f1, label="val macro F1", linewidth=1.5, linestyle="--")
    axes[1].set_title("Accuracy / F1 curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"

    assets, train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        imbalance_correction=args.imbalance_correction,
        input_mode=args.input_mode,
        image_size=args.image_size,
    )

    experiment = GrainCNNExperiment()
    params = count_parameters(experiment.model)

    with log_path.open("w", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(experiment.get_model_summary(args.image_size))
        log(f"Total parameters: {params['total']:,}")
        log(f"Trainable parameters: {params['trainable']:,}")
        log(f"Class weights: {json.dumps(assets.class_weights, sort_keys=True)}")
        pos_weight = assets.pos_weight if args.imbalance_correction == "loss" else 1.0
        log(f"Imbalance correction: {args.imbalance_correction}")
        log(f"Input mode: {args.input_mode}")
        log(f"Image size: {args.image_size}x{args.image_size}")
        log(f"Positive-class weight (equiaxed): {pos_weight:.4f}")
        log(f"Train / val / test sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")

        history = experiment.train(
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            pos_weight=pos_weight,
            patience=args.patience,
            reduce_lr_patience=args.reduce_lr_patience,
            checkpoint_path=output_dir / "best_model.pt",
            history_path=None,
            input_size=args.image_size,
            log_fn=log,
        )

        save_history(history, output_dir / "history.csv")
        plot_history(history, output_dir / "training_curves.png")

        calibration_metrics = experiment.evaluate(val_loader)
        decision_threshold, threshold_metrics = find_best_threshold(
            np.asarray(calibration_metrics["y_true"], dtype=np.int64),
            np.asarray(calibration_metrics["y_prob"], dtype=np.float32),
            metric="macro_f1",
        )
        val_metrics = experiment.evaluate(val_loader, threshold=decision_threshold)
        test_metrics = experiment.evaluate(test_loader, threshold=decision_threshold)
        best_history_row = min(history, key=lambda row: row["val_loss"]) if history else {"epoch": 0, "val_loss": float("nan")}
        metrics_payload = {
            "dataset_root": str(Path(args.dataset_root)),
            "split_dir": str(Path(args.split_dir)),
            "input_mode": args.input_mode,
            "image_size": args.image_size,
            "imbalance_correction": args.imbalance_correction,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history),
            "best_epoch": int(best_history_row["epoch"]),
            "best_val_loss": float(best_history_row["val_loss"]),
            "decision_threshold": float(decision_threshold),
            "threshold_calibration": threshold_metrics,
            "history": history,
            "validation": val_metrics,
            "test": test_metrics,
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary = [
            "# Grain CNN training summary",
            "",
            f"- Train / val / test sizes: {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}",
            f"- Best checkpoint: {output_dir / 'best_model.pt'}",
            f"- Training curves: {output_dir / 'training_curves.png'}",
            f"- Best validation loss: {best_history_row['val_loss']:.4f}",
            f"- Best validation epoch: {int(best_history_row['epoch'])}",
            f"- Decision threshold: {decision_threshold:.3f}",
            f"- Validation accuracy at final checkpoint: {val_metrics['accuracy']:.4f}",
            f"- Final test accuracy: {test_metrics['accuracy']:.4f}",
            f"- Final test macro F1: {test_metrics['macro_f1']:.4f}",
            f"- Final test weighted F1: {test_metrics['weighted_f1']:.4f}",
        ]
        (output_dir / "results_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        log(json.dumps(metrics_payload, indent=2, sort_keys=True))
        log(f"Saved training curves: {output_dir / 'training_curves.png'}")
        log(f"Saved metrics: {output_dir / 'metrics.json'}")
        log(f"Saved summary: {output_dir / 'results_summary.md'}")


if __name__ == "__main__":
    main()
