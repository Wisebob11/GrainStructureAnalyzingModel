from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetLayoutError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class DatasetLayout:
    dataset_root: Path
    image_root: Path
    metadata_csv: Path


def resolve_dataset_layout(dataset_root: str | Path) -> DatasetLayout:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise DatasetLayoutError(
            f"Dataset root not found: {root}. Check the folder name/location."
        )

    metadata_csv = root / "all_grains.csv"
    if not metadata_csv.exists():
        raise DatasetLayoutError(
            f"Expected metadata CSV at {metadata_csv}, but it was not found. "
            "Check the dataset folder name/location."
        )

    direct_class_root = root if _has_class_subdirs(root) else None
    nested_class_root = root / "grain_crops" if _has_class_subdirs(root / "grain_crops") else None

    image_root = direct_class_root or nested_class_root
    if image_root is None:
        raise DatasetLayoutError(
            "Could not find the required image folders. Expected either:\n"
            f"  - {root / 'columnar'} and {root / 'equiaxed'}\n"
            f"  - or {root / 'grain_crops' / 'columnar'} and {root / 'grain_crops' / 'equiaxed'}\n"
            "Check the folder name/location and do not rely on unrelated files in the dataset root."
        )

    return DatasetLayout(dataset_root=root, image_root=image_root, metadata_csv=metadata_csv)


def _has_class_subdirs(path: Path) -> bool:
    return (path / "columnar").is_dir() and (path / "equiaxed").is_dir()


def read_csv_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv_rows(csv_path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sanitize_source_prefix(source_image: str) -> str:
    """Derive the file-name prefix used by the crop images from the source image name."""
    name = Path(source_image).name
    for ext in sorted(ALLOWED_IMAGE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    stem = name.replace(".", "")
    stem = re.split(
        r"(?i)(?:_IPF.*|_EBSD.*|_EBSDMAP.*|_MAP.*|_MONTAGE.*|_Grain.*|_Structure.*|_Image.*)",
        stem,
        maxsplit=1,
    )[0]
    stem = re.sub(r"[^A-Za-z0-9_]+", "", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse float from {value!r}") from exc


def allowed_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_SUFFIXES


def find_crop_path(image_root: Path, row: dict[str, str]) -> Path:
    """Find the crop image for a CSV row.

    This first tries the deterministic prefix/grain-id/aspect-ratio pattern used by
    the segmentation pipeline, then falls back to a source-specific glob only within
    the known class folder.
    """
    label = row["morphology_label"]
    class_dir = image_root / label
    if not class_dir.is_dir():
        raise DatasetLayoutError(f"Missing class folder: {class_dir}")

    grain_id = int(float(row["grain_id"]))
    aspect_ratio = float(row["aspect_ratio"])
    prefix = sanitize_source_prefix(row["source_image"])
    expected_stem = f"{prefix}_grain_{grain_id:04d}_ar{aspect_ratio:.2f}"

    for suffix in ALLOWED_IMAGE_SUFFIXES:
        candidate = class_dir / f"{expected_stem}{suffix}"
        if candidate.exists():
            return candidate

    # Fallback: search only inside the relevant class folder and only for image files.
    candidates: list[Path] = []
    glob_pattern = f"*grain_{grain_id:04d}_ar{aspect_ratio:.2f}.*"
    for path in class_dir.glob(glob_pattern):
        if allowed_image_file(path):
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # Deterministic tie-breaker.
        candidates.sort()
        return candidates[0]

    # Last fallback: search by grain id only, still restricted to the class folder.
    loose = [p for p in class_dir.glob(f"*grain_{grain_id:04d}.*") if allowed_image_file(p)]
    if len(loose) == 1:
        return loose[0]
    if len(loose) > 1:
        loose.sort()
        return loose[0]

    raise DatasetLayoutError(
        "Could not match a crop image for this CSV row. "
        f"Looked in {class_dir} for prefix {prefix!r}, grain_id {grain_id}, aspect_ratio {aspect_ratio:.2f}."
    )


def load_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    return read_csv_rows(manifest_path)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
