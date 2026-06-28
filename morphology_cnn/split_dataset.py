from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from common import (
    DatasetLayoutError,
    find_crop_path,
    read_csv_rows,
    resolve_dataset_layout,
    save_csv_rows,
    sanitize_source_prefix,
    write_json,
)

MANIFEST_FIELDS = [
    "image_path",
    "label",
    "source_image",
    "grain_id",
    "aspect_ratio",
    "confidence_tier",
    "split",
]


def choose_test_sources(rows: list[dict[str, str]], test_source: str | None) -> list[str]:
    sources = sorted({row["source_image"] for row in rows})
    if not sources:
        raise DatasetLayoutError("No source images were found in the metadata CSV.")

    if test_source and test_source.lower() not in {"auto", "smallest", "largest"}:
        if test_source not in sources:
            raise DatasetLayoutError(
                f"Requested test source {test_source!r} was not found in the metadata CSV. "
                f"Available sources: {sources}"
            )
        return [test_source]

    counts = Counter(row["source_image"] for row in rows)
    if test_source and test_source.lower() == "largest":
        selected = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return [selected]
    # Default: hold out the smallest source so the train set stays as large as possible.
    selected = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [selected]


def build_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    val_fraction: float,
    seed: int,
    test_source: str | None,
) -> dict:
    layout = resolve_dataset_layout(dataset_root)
    rows = read_csv_rows(layout.metadata_csv)

    test_sources = choose_test_sources(rows, test_source)
    trainval_rows = [row for row in rows if row["source_image"] not in test_sources]
    test_rows = [row for row in rows if row["source_image"] in test_sources]

    if not trainval_rows:
        raise DatasetLayoutError(
            "No training rows remain after the source-level split. "
            "Check the dataset layout and the requested test source."
        )
    if not test_rows:
        raise DatasetLayoutError(
            "No test rows were selected. Check the dataset layout and the requested test source."
        )

    stratify = [row["morphology_label"] for row in trainval_rows]
    if len(set(stratify)) < 2:
        stratify = None

    train_rows, val_rows = train_test_split(
        trainval_rows,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )

    # Deterministic order inside each manifest.
    train_rows = sorted(train_rows, key=lambda r: (r["source_image"], int(float(r["grain_id"]))))
    val_rows = sorted(val_rows, key=lambda r: (r["source_image"], int(float(r["grain_id"]))))
    test_rows = sorted(test_rows, key=lambda r: (r["source_image"], int(float(r["grain_id"]))))

    def to_manifest_row(row: dict[str, str], split: str) -> dict[str, str]:
        crop_path = find_crop_path(layout.image_root, row)
        return {
            "image_path": str(crop_path.relative_to(layout.dataset_root)),
            "label": row["morphology_label"],
            "source_image": row["source_image"],
            "grain_id": row["grain_id"],
            "aspect_ratio": row["aspect_ratio"],
            "confidence_tier": row.get("confidence_tier", ""),
            "split": split,
        }

    train_manifest = [to_manifest_row(row, "train") for row in train_rows]
    val_manifest = [to_manifest_row(row, "val") for row in val_rows]
    test_manifest = [to_manifest_row(row, "test") for row in test_rows]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv_rows(output_dir / "train_manifest.csv", train_manifest, MANIFEST_FIELDS)
    save_csv_rows(output_dir / "val_manifest.csv", val_manifest, MANIFEST_FIELDS)
    save_csv_rows(output_dir / "test_manifest.csv", test_manifest, MANIFEST_FIELDS)

    summary = {
        "dataset_root": str(layout.dataset_root),
        "image_root": str(layout.image_root),
        "metadata_csv": str(layout.metadata_csv),
        "seed": seed,
        "val_fraction": val_fraction,
        "test_sources": test_sources,
        "train_sources": sorted({row["source_image"] for row in train_rows}),
        "counts": {
            "train": len(train_manifest),
            "val": len(val_manifest),
            "test": len(test_manifest),
            "all": len(rows),
        },
        "class_counts": {
            "train": dict(Counter(row["label"] for row in train_manifest)),
            "val": dict(Counter(row["label"] for row in val_manifest)),
            "test": dict(Counter(row["label"] for row in test_manifest)),
        },
        "source_counts": {
            "train": dict(Counter(row["source_image"] for row in train_manifest)),
            "val": dict(Counter(row["source_image"] for row in val_manifest)),
            "test": dict(Counter(row["source_image"] for row in test_manifest)),
        },
        "source_prefixes": {
            source: sanitize_source_prefix(source)
            for source in sorted({row["source_image"] for row in rows})
        },
    }
    write_json(output_dir / "split_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible source-level train/val/test manifests.")
    parser.add_argument("--dataset-root", type=str, default="Dataset", help="Path to the dataset root.")
    parser.add_argument("--output-dir", type=str, default="artifacts/splits", help="Where to save manifest CSVs.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation fraction carved from the training sources.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the validation split.")
    parser.add_argument(
        "--test-source",
        type=str,
        default="auto",
        help="Specific source_image to hold out for test, or auto/smallest/largest.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = build_manifests(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        test_source=args.test_source,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
