from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from morphology_cnn.common import CLASS_NAMES, LABEL_TO_INT, DatasetLayoutError, load_manifest, resolve_dataset_layout

IMAGE_SIZE = 256
POSITIVE_CLASS = "equiaxed"


@dataclass(frozen=True)
class SplitAssets:
    train_rows: list[dict[str, str]]
    val_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]
    class_weights: dict[str, float]
    pos_weight: float


class GrainPatchDataset(Dataset):
    def __init__(self, dataset_root: str | Path, manifest_path: str | Path, transform=None):
        self.dataset_root = resolve_dataset_layout(dataset_root).dataset_root
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.rows = load_manifest(self.manifest_path)
        if not self.rows:
            raise DatasetLayoutError(f"Manifest is empty: {self.manifest_path}")
        self.transform = transform
        self.label_to_int = LABEL_TO_INT

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_absolute():
            image_path = self.dataset_root / image_path
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(self.label_to_int[row["label"]], dtype=torch.float32)
        return image, label, row


def build_transforms(train: bool, input_mode: str = "rgb", image_size: int = IMAGE_SIZE) -> transforms.Compose:
    color_transform = transforms.Grayscale(num_output_channels=3) if input_mode == "shape" else transforms.Lambda(lambda image: image)
    if train:
        return transforms.Compose(
            [
                transforms.Lambda(lambda image: ImageOps.pad(image, (image_size, image_size), method=Image.Resampling.BILINEAR, color=0)),
                color_transform,
                transforms.RandomRotation(degrees=(0, 360)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Lambda(lambda image: ImageOps.pad(image, (image_size, image_size), method=Image.Resampling.BILINEAR, color=0)),
            color_transform,
            transforms.ToTensor(),
        ]
    )


def read_split_rows(split_dir: str | Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    split_dir = Path(split_dir)
    train_rows = load_manifest(split_dir / "train_manifest.csv")
    val_rows = load_manifest(split_dir / "val_manifest.csv")
    test_rows = load_manifest(split_dir / "test_manifest.csv")
    return train_rows, val_rows, test_rows


def compute_class_stats(rows: Iterable[dict[str, str]]) -> tuple[dict[str, float], float]:
    counts = Counter(row["label"] for row in rows)
    total = sum(counts.values())
    if counts[POSITIVE_CLASS] == 0:
        raise DatasetLayoutError("Training split contains no equiaxed examples, cannot compute pos_weight.")
    class_weights = {
        label: total / (len(counts) * count)
        for label, count in counts.items()
    }
    pos_weight = counts["columnar"] / counts["equiaxed"]
    return class_weights, pos_weight


def make_weighted_sampler(rows: list[dict[str, str]]) -> WeightedRandomSampler:
    counts = Counter(row["label"] for row in rows)
    sample_weights = [1.0 / counts[row["label"]] for row in rows]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_assets(dataset_root: str | Path, split_dir: str | Path) -> SplitAssets:
    layout = resolve_dataset_layout(dataset_root)
    train_rows, val_rows, test_rows = read_split_rows(split_dir)
    class_weights, pos_weight = compute_class_stats(train_rows)
    return SplitAssets(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        class_weights=class_weights,
        pos_weight=pos_weight,
    )


def build_dataloaders(
    dataset_root: str | Path,
    split_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    imbalance_correction: str = "sampler",
    input_mode: str = "rgb",
    image_size: int = IMAGE_SIZE,
):
    layout = resolve_dataset_layout(dataset_root)
    assets = build_assets(layout.dataset_root, split_dir)

    train_dataset = GrainPatchDataset(layout.dataset_root, Path(split_dir) / "train_manifest.csv", transform=build_transforms(True, input_mode, image_size))
    val_dataset = GrainPatchDataset(layout.dataset_root, Path(split_dir) / "val_manifest.csv", transform=build_transforms(False, input_mode, image_size))
    test_dataset = GrainPatchDataset(layout.dataset_root, Path(split_dir) / "test_manifest.csv", transform=build_transforms(False, input_mode, image_size))

    sampler = make_weighted_sampler(assets.train_rows) if imbalance_correction == "sampler" else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return assets, train_loader, val_loader, test_loader
