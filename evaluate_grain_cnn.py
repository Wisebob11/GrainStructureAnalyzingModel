from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageOps

from grain_cnn_model import CLASS_NAMES, GrainCNNExperiment
from grain_data_loader import GrainPatchDataset, build_transforms, read_split_rows
from morphology_cnn.common import AMBIGUOUS_MAX_AR, AMBIGUOUS_MIN_AR, resolve_dataset_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained custom grain CNN.")
    parser.add_argument("--dataset-root", type=str, default="Dataset_merged")
    parser.add_argument("--split-dir", type=str, default="artifacts/splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/grain_cnn_run/eval")
    parser.add_argument("--source-image", type=str, default="Dr.Lu_Grain_Structure_Image")
    parser.add_argument("--threshold", type=float, default=None, help="Override the validation-tuned threshold.")
    parser.add_argument("--input-mode", choices=("rgb", "shape"), default="rgb")
    parser.add_argument("--image-size", type=int, choices=(64, 128, 256), default=256)
    return parser.parse_args()


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_breakdown(detail: dict[str, object], output_path: Path) -> None:
    bins = detail["aspect_ratio_bins"]
    names = list(bins)
    accuracies = [float(bins[name]["accuracy"]) for name in names]
    sizes = [int(bins[name]["n"]) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars = axes[0].bar(names, accuracies, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy by aspect-ratio bin")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, accuracy, size in zip(bars, accuracies, sizes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, accuracy + 0.03, f"{accuracy:.1%}\n(n={size})", ha="center")

    ambiguity = detail["ambiguous"]
    labels = ["All grains", "Without ambiguous"]
    values = [
        float(ambiguity["with_ambiguous"]["accuracy"]),
        float(ambiguity["without_ambiguous"]["accuracy"]),
    ]
    bars = axes[1].bar(labels, values, color=["#B279A2", "#E45756"])
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Ambiguous band ({AMBIGUOUS_MIN_AR}-{AMBIGUOUS_MAX_AR})")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.1%}", ha="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, color in zip((0, 1), ("#4C78A8", "#E45756")):
        ax.hist(y_prob[y_true == label], bins=np.linspace(0, 1, 21), alpha=0.65, label=CLASS_NAMES[label], color=color)
    ax.set_xlabel("Predicted equiaxed probability")
    ax.set_ylabel("Number of grains")
    ax.set_title("Prediction confidence by true class")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_tile(image: Image.Image, text_lines: list[str]) -> Image.Image:
    tile = image.copy().convert("RGB")
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    overlay_height = 48 + 16 * max(0, len(text_lines) - 2)
    draw.rectangle([0, tile.height - overlay_height, tile.width, tile.height], fill=(0, 0, 0))
    y = tile.height - overlay_height + 6
    for line in text_lines:
        draw.text((6, y), line, fill=(255, 255, 255), font=font)
        y += 16
    return tile


def save_example_grid(examples: list[dict[str, object]], output_path: Path) -> None:
    if not examples:
        return
    tiles = []
    for example in examples:
        image = Image.open(example["path"]).convert("RGB")
        image = ImageOps.pad(image, (256, 256), method=Image.Resampling.BILINEAR, color=0)
        tiles.append(
            annotate_tile(
                image,
                [
                    f"True: {example['true_label']}",
                    f"Pred: {example['pred_label']} ({example['confidence']:.2f})",
                ],
            )
        )

    cols = 4
    rows = math.ceil(len(tiles) / cols)
    canvas = Image.new("RGB", (cols * 256, rows * 256), (18, 18, 18))
    for index, tile in enumerate(tiles):
        x = (index % cols) * 256
        y = (index // cols) * 256
        canvas.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [centre - margin, centre + margin]


def accuracy_report(y_true: np.ndarray, y_pred: np.ndarray, rows: list[dict[str, object]]) -> dict[str, object]:
    def report(indices: np.ndarray) -> dict[str, object]:
        correct = int(np.sum(y_true[indices] == y_pred[indices]))
        return {"n": int(len(indices)), "accuracy": correct / len(indices) if len(indices) else float("nan"),
                "accuracy_95_ci": wilson_interval(correct, len(indices))}

    ar = np.asarray([float(row["aspect_ratio"]) for row in rows])
    ambiguous = (ar >= AMBIGUOUS_MIN_AR) & (ar <= AMBIGUOUS_MAX_AR)
    bins = {"<1.7": ar < 1.7, "1.7-2.3": ambiguous, ">2.3": ar > 2.3}
    return {
        "per_class_recall": {
            CLASS_NAMES[index]: float(np.sum((y_true == index) & (y_pred == index)) / max(1, np.sum(y_true == index)))
            for index in range(2)
        },
        "aspect_ratio_bins": {name: report(np.flatnonzero(mask)) for name, mask in bins.items()},
        "ambiguous": {
            "range": [AMBIGUOUS_MIN_AR, AMBIGUOUS_MAX_AR],
            "with_ambiguous": report(np.arange(len(rows))),
            "without_ambiguous": report(np.flatnonzero(~ambiguous)),
        },
    }


def choose_examples(records: list[dict[str, object]], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, k: int = 8):
    scored = []
    for idx, record in enumerate(records):
        correct = int(y_true[idx]) == int(y_pred[idx])
        confidence = y_prob[idx] if int(y_pred[idx]) == 1 else 1.0 - y_prob[idx]
        scored.append((correct, confidence, idx, record))

    correct_sorted = [item for item in scored if item[0]]
    incorrect_sorted = [item for item in scored if not item[0]]
    correct_sorted.sort(key=lambda item: item[1], reverse=True)
    incorrect_sorted.sort(key=lambda item: item[1])

    chosen = incorrect_sorted[: min(4, len(incorrect_sorted))] + correct_sorted[: max(0, k - min(4, len(incorrect_sorted)))]
    if len(chosen) < k:
        remaining = [item for item in scored if item not in chosen]
        remaining.sort(key=lambda item: abs(item[1] - 0.5))
        chosen.extend(remaining[: k - len(chosen)])

    examples = []
    for correct, confidence, idx, record in chosen[:k]:
        examples.append(
            {
                "path": record["image_path"],
                "true_label": CLASS_NAMES[int(y_true[idx])],
                "pred_label": CLASS_NAMES[int(y_pred[idx])],
                "confidence": float(confidence),
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = GrainCNNExperiment.load_model(args.checkpoint)
    layout = resolve_dataset_layout(args.dataset_root)
    test_dataset = GrainPatchDataset(args.dataset_root, Path(args.split_dir) / "test_manifest.csv", transform=build_transforms(False, args.input_mode, args.image_size))

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    _, val_rows, test_rows = read_split_rows(args.split_dir)
    val_dataset = GrainPatchDataset(args.dataset_root, Path(args.split_dir) / "val_manifest.csv", transform=build_transforms(False, args.input_mode, args.image_size))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    val_metrics = experiment.evaluate(val_loader)
    from grain_cnn_model import find_best_threshold
    tuned_threshold, threshold_metrics = find_best_threshold(
        np.asarray(val_metrics["y_true"], dtype=np.int64),
        np.asarray(val_metrics["y_prob"], dtype=np.float32),
    )
    threshold = args.threshold if args.threshold is not None else tuned_threshold
    metrics = experiment.evaluate(loader, threshold=threshold)
    y_true = np.asarray(metrics["y_true"], dtype=np.int64)
    y_prob = np.asarray(metrics["y_prob"], dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)
    metrics["threshold"] = threshold
    metrics["threshold_source"] = "cli" if args.threshold is not None else "validation_macro_f1"
    metrics["threshold_calibration"] = threshold_metrics
    metrics["evaluation_detail"] = accuracy_report(y_true, y_pred, test_rows)
    metrics["fixed_0_5"] = experiment.evaluate(loader, threshold=0.5)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_evaluation_breakdown(metrics["evaluation_detail"], output_dir / "evaluation_breakdown.png")
    plot_confidence_distribution(y_true, y_prob, output_dir / "confidence_distribution.png")
    examples = choose_examples(test_rows, y_true, y_pred, y_prob, k=8)
    for example in examples:
        example["path"] = str(layout.dataset_root / example["path"])
    save_example_grid(examples, output_dir / "example_predictions.png")

    source_records = [row for row in test_rows if row["source_image"] == args.source_image]
    source_indices = [idx for idx, row in enumerate(test_rows) if row["source_image"] == args.source_image]
    source_probs = y_prob[source_indices]
    source_preds = y_pred[source_indices]
    source_true = y_true[source_indices]
    source_report = {
        "source_image": args.source_image,
        "num_patches": len(source_indices),
        "true_columnar_fraction": float(np.mean(source_true == 0)),
        "predicted_columnar_fraction": float(np.mean(source_preds == 0)),
        "mean_equiaxed_probability": float(np.mean(source_probs)) if source_probs.size else float("nan"),
        "majority_vote_label": CLASS_NAMES[int(np.round(np.mean(source_preds)))] if source_indices else None,
        "source_accuracy": float(np.mean(source_true == source_preds)) if source_indices else None,
        "note": f"Patch-level aggregation over the Dr. Lu source image, because the classifier operates on {args.image_size}x{args.image_size} grain patches.",
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "source_report.json").write_text(json.dumps(source_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# Grain CNN evaluation summary",
        "",
        f"- Test accuracy at threshold {threshold:.3f}: {metrics['accuracy']:.4f}",
        f"- Test accuracy at threshold 0.500: {metrics['fixed_0_5']['accuracy']:.4f}",
        f"- 95% accuracy CI: {metrics['evaluation_detail']['ambiguous']['with_ambiguous']['accuracy_95_ci']}",
        f"- Columnar recall: {metrics['evaluation_detail']['per_class_recall']['columnar']:.4f}",
        f"- Evaluation breakdown: {output_dir / 'evaluation_breakdown.png'}",
        f"- Confidence distribution: {output_dir / 'confidence_distribution.png'}",
        f"- Test macro F1: {metrics['macro_f1']:.4f}",
        f"- Test weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Confusion matrix: {cm.tolist()}",
        f"- Example grid: {output_dir / 'example_predictions.png'}",
        f"- Dr. Lu source report: {output_dir / 'source_report.json'}",
    ]
    (output_dir / "results_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(source_report, indent=2, sort_keys=True))
    print(f"Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved example grid: {output_dir / 'example_predictions.png'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved source report: {output_dir / 'source_report.json'}")


if __name__ == "__main__":
    main()
