from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from grain_cnn_model import CLASS_NAMES, GrainCNNExperiment
from grain_data_loader import GrainPatchDataset, build_transforms, read_split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained custom grain CNN.")
    parser.add_argument("--dataset-root", type=str, default="Dataset")
    parser.add_argument("--split-dir", type=str, default="artifacts/splits")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="artifacts/grain_cnn_run/eval")
    parser.add_argument("--source-image", type=str, default="Dr.Lu_Grain_Structure_Image")
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


def annotate_tile(image: Image.Image, text_lines: list[str]) -> Image.Image:
    tile = image.copy().convert("RGB")
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except Exception:
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
        image = image.resize((256, 256))
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
    _, _, test_rows = read_split_rows(args.split_dir)
    test_dataset = GrainPatchDataset(args.dataset_root, Path(args.split_dir) / "test_manifest.csv", transform=build_transforms(False))

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    metrics = experiment.evaluate(loader)
    y_true = np.asarray(metrics["y_true"], dtype=np.int64)
    y_prob = np.asarray(metrics["y_prob"], dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    cm = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    examples = choose_examples(test_rows, y_true, y_pred, y_prob, k=8)
    for example in examples:
        example["path"] = str(Path(args.dataset_root) / example["path"])
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
        "mean_equiaxed_probability": float(np.mean(source_probs)),
        "majority_vote_label": CLASS_NAMES[int(np.round(np.mean(source_preds)))],
        "source_accuracy": float(np.mean(source_true == source_preds)),
        "note": "Patch-level aggregation over the Dr. Lu source image, because the classifier operates on 256x256 grain patches.",
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "source_report.json").write_text(json.dumps(source_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = [
        "# Grain CNN evaluation summary",
        "",
        f"- Test accuracy: {metrics['accuracy']:.4f}",
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
