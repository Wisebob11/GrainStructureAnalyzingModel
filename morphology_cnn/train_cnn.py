from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from common import DatasetLayoutError, load_manifest, resolve_dataset_layout, write_json

CLASS_NAMES = ["columnar", "equiaxed"]
IMAGE_SIZE = 256


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GrainManifestDataset(Dataset):
    def __init__(self, dataset_root: str | Path, manifest_path: str | Path, transform=None):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.rows = load_manifest(self.manifest_path)
        self.transform = transform
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.dataset_root / image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, label


def build_transforms(train: bool) -> transforms.Compose:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=180, fill=0),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.20,
                            contrast=0.20,
                            saturation=0.08,
                            hue=0.01,
                        )
                    ],
                    p=0.8,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_model(num_classes: int = 2) -> nn.Module:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights ({e}), using random initialization. "
              "The model will require more epochs to train.")
        model = efficientnet_b0(weights=None)
    
    if model is None:
        model = efficientnet_b0(weights=None)
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_targets: list[int] = []
    all_probs: list[np.ndarray] = []
    all_preds: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader.dataset))
    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_preds)
    y_prob = np.asarray(all_probs)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_macro_f1 = [row["train_macro_f1"] for row in history]
    val_macro_f1 = [row["val_macro_f1"] for row in history]
    train_acc = [row["train_accuracy"] for row in history]
    val_acc = [row["val_accuracy"] for row in history]
    val_balanced_acc = [row["val_balanced_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, marker="o", linewidth=2, label="train loss")
    axes[0].plot(epochs, val_loss, marker="o", linewidth=2, label="val loss")
    axes[0].set_title("Loss over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_macro_f1, marker="o", linewidth=2, label="train macro F1")
    axes[1].plot(epochs, val_macro_f1, marker="o", linewidth=2, label="val macro F1")
    axes[1].plot(epochs, train_acc, marker="s", linewidth=1.6, linestyle="--", label="train accuracy")
    axes[1].plot(epochs, val_acc, marker="s", linewidth=1.6, linestyle="--", label="val accuracy")
    axes[1].plot(epochs, val_balanced_acc, marker="^", linewidth=1.6, linestyle=":", label="val balanced acc")
    axes[1].set_title("Classification metrics over epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_weighted_sampler(rows: list[dict[str, str]]) -> tuple[WeightedRandomSampler, torch.Tensor]:
    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    targets = np.asarray([label_to_idx[row["label"]] for row in rows], dtype=np.int64)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray([0, 1]),
        y=targets,
    )
    sample_weights = np.asarray([class_weights[t] for t in targets], dtype=np.float64)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, torch.as_tensor(class_weights, dtype=torch.float32)


def load_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    rows = load_manifest(manifest_path)
    if not rows:
        raise DatasetLayoutError(f"Manifest is empty: {manifest_path}")
    return rows


def save_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a transfer-learning CNN for grain morphology classification.")
    parser.add_argument("--dataset-root", type=str, default="Dataset", help="Dataset root used to resolve manifest paths.")
    parser.add_argument("--split-dir", type=str, default="artifacts/splits", help="Directory containing train/val/test manifests.")
    parser.add_argument("--output-dir", type=str, default="artifacts/model_run", help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=18, help="Maximum number of epochs.")
    parser.add_argument("--head-epochs", type=int, default=3, help="Number of warm-up epochs with the backbone frozen.")
    parser.add_argument("--batch-size", type=int, default=24, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for fine-tuning.")
    parser.add_argument("--head-lr", type=float, default=8e-4, help="Learning rate during the frozen-backbone warm-up stage.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on validation macro F1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    layout = resolve_dataset_layout(args.dataset_root)
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(split_dir / "train_manifest.csv")
    val_rows = load_rows(split_dir / "val_manifest.csv")
    test_rows = load_rows(split_dir / "test_manifest.csv")

    train_dataset = GrainManifestDataset(layout.dataset_root, split_dir / "train_manifest.csv", transform=build_transforms(True))
    val_dataset = GrainManifestDataset(layout.dataset_root, split_dir / "val_manifest.csv", transform=build_transforms(False))
    test_dataset = GrainManifestDataset(layout.dataset_root, split_dir / "test_manifest.csv", transform=build_transforms(False))

    sampler, class_weights = make_weighted_sampler(train_rows)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_path = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    best_macro_f1 = -math.inf
    best_epoch = -1
    epochs_without_improve = 0

    def make_optimizer(lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=args.weight_decay)

    # Stage 1: warm-up the classification head.
    set_backbone_trainable(model, False)
    optimizer = make_optimizer(args.head_lr)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.head_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(args.lr)

        model.train()
        running_loss = 0.0
        all_train_targets: list[int] = []
        all_train_preds: list[int] = []

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_train_targets.extend(targets.detach().cpu().tolist())
            all_train_preds.extend(preds.detach().cpu().tolist())

        train_loss = running_loss / max(1, len(train_dataset))
        train_macro_f1 = f1_score(all_train_targets, all_train_preds, average="macro", zero_division=0)
        train_acc = accuracy_score(all_train_targets, all_train_preds)

        val_metrics = run_eval(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_macro_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.4f} | train macro F1 {train_macro_f1:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f} | "
            f"val bal acc {val_metrics['balanced_accuracy']:.4f} | val macro F1 {val_metrics['macro_f1']:.4f} | "
            f"val weighted F1 {val_metrics['weighted_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "dataset_root": str(layout.dataset_root),
                    "split_dir": str(split_dir),
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping after {epoch} epochs; best validation macro F1 was {best_macro_f1:.4f} at epoch {best_epoch}.")
                break

    if not best_path.exists():
        raise RuntimeError("Training finished without saving a checkpoint.")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    val_metrics = run_eval(model, val_loader, device)
    test_metrics = run_eval(model, test_loader, device)

    cm = np.asarray(test_metrics["confusion_matrix"], dtype=np.int64)
    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    report = {
        "dataset_root": str(layout.dataset_root),
        "split_dir": str(split_dir),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "validation": {
            "loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "balanced_accuracy": val_metrics["balanced_accuracy"],
            "macro_f1": val_metrics["macro_f1"],
            "weighted_f1": val_metrics["weighted_f1"],
            "classification_report": val_metrics["classification_report"],
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
        "test": {
            "loss": test_metrics["loss"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_f1": test_metrics["weighted_f1"],
            "classification_report": test_metrics["classification_report"],
            "confusion_matrix": test_metrics["confusion_matrix"],
        },
        "train_sources": sorted({row["source_image"] for row in train_rows}),
        "val_sources": sorted({row["source_image"] for row in val_rows}),
        "test_sources": sorted({row["source_image"] for row in test_rows}),
        "train_counts": dict(Counter(row["label"] for row in train_rows)),
        "val_counts": dict(Counter(row["label"] for row in val_rows)),
        "test_counts": dict(Counter(row["label"] for row in test_rows)),
    }

    write_json(output_dir / "metrics.json", report)
    save_history(output_dir / "history.csv", history)

    summary_lines = [
        "# Grain morphology CNN results",
        "",
        "This is a proof-of-concept transfer-learning model, not a production estimate.",
        "The held-out test set is source-image separated to avoid patch leakage.",
        "",
        f"- Train sources: {', '.join(report['train_sources'])}",
        f"- Validation sources: {', '.join(report['val_sources'])}",
        f"- Test sources: {', '.join(report['test_sources'])}",
        f"- Best validation macro F1: {best_macro_f1:.4f} at epoch {best_epoch}",
        f"- Test accuracy: {test_metrics['accuracy']:.4f}",
        f"- Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}",
        f"- Test macro F1: {test_metrics['macro_f1']:.4f}",
        "",
        "Confusion matrix (rows=true, columns=predicted):",
        "",
        f"{cm.tolist()}",
        "",
        "Per-class metrics are stored in artifacts/model_run/metrics.json.",
        "",
        "Interpretation note: morphology-only inference can distinguish columnar vs. equiaxed appearance,",
        "but it cannot produce calibrated mechanical-property numbers without a separate dataset that pairs",
        "grain statistics with measured properties for the same alloy and processing state.",
    ]
    (output_dir / "results_summary.md").write_text("\n".join(summary_lines) + "\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved training curves: {output_dir / 'training_curves.png'}")
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved summary: {output_dir / 'results_summary.md'}")


if __name__ == "__main__":
    main()
