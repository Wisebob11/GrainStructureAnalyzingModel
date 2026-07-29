from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support

CLASS_NAMES = ["columnar", "equiaxed"]


def _conv_block(in_channels: int, out_channels: int, convs: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_in = in_channels
    for _ in range(convs):
        layers.extend(
            [
                nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        current_in = out_channels
    layers.extend(
        [
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        ]
    )
    return nn.Sequential(*layers)


class GrainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32, convs=2, dropout=0.25),
            _conv_block(32, 64, convs=2, dropout=0.25),
            _conv_block(64, 128, convs=2, dropout=0.30),
            _conv_block(128, 256, convs=1, dropout=0.30),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.60),
            nn.Linear(256, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)
        return outputs.squeeze(1)


def build_model() -> GrainCNN:
    return GrainCNN()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def get_model_summary(model: nn.Module, input_shape: tuple[int, int, int, int] = (1, 3, 256, 256)) -> str:
    params = count_parameters(model)
    lines = [
        "GrainCNN summary",
        f"Input shape: {input_shape}",
        f"Total parameters: {params['total']:,}",
        f"Trainable parameters: {params['trainable']:,}",
        "",
        repr(model),
    ]
    return "\n".join(lines)


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "y_pred": y_pred.tolist(),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "macro_f1") -> tuple[float, dict[str, Any]]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    best_score = float("-inf")

    for threshold in thresholds:
        metrics = _binary_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics.get(metric, float("-inf")))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        best_metrics = _binary_metrics(y_true, y_prob, threshold=best_threshold)

    return best_threshold, best_metrics


class GrainCNNExperiment:
    def __init__(self, model: nn.Module | None = None, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or build_model()).to(self.device)

    def get_model_summary(self) -> str:
        return get_model_summary(self.model)

    def train(
        self,
        train_loader,
        val_loader,
        *,
        epochs: int = 60,
        lr: float = 1e-3,
        pos_weight: float = 1.0,
        patience: int = 10,
        reduce_lr_patience: int = 5,
        factor: float = 0.5,
        checkpoint_path: str | Path | None = None,
        history_path: str | Path | None = None,
        log_fn=print,
    ) -> list[dict[str, float]]:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=reduce_lr_patience,
        )

        best_state = None
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improve = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0.0
            train_targets: list[int] = []
            train_probs: list[float] = []

            for images, targets, _meta in train_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                train_targets.extend(targets.detach().cpu().numpy().astype(int).tolist())
                train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

            train_loss /= max(1, len(train_loader.dataset))
            train_metrics = _binary_metrics(np.asarray(train_targets), np.asarray(train_probs))

            val_metrics = self.evaluate(val_loader, criterion=criterion)
            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            history_row = {
                "epoch": float(epoch),
                "lr": float(current_lr),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_weighted_f1": float(val_metrics["weighted_f1"]),
            }
            history.append(history_row)

            log_fn(
                f"Epoch {epoch:02d}/{epochs} | lr {current_lr:.2e} | "
                f"train loss {train_loss:.4f} | train acc {train_metrics['accuracy']:.4f} | train macro F1 {train_metrics['macro_f1']:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | val acc {val_metrics['accuracy']:.4f} | "
                f"val macro F1 {val_metrics['macro_f1']:.4f} | val weighted F1 {val_metrics['weighted_f1']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss - 1e-5:
                best_val_loss = float(val_metrics["loss"])
                best_epoch = epoch
                epochs_without_improve = 0
                best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "input_shape": [3, 256, 256],
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                }
                if checkpoint_path is not None:
                    self.save_model(checkpoint_path, metadata=best_state)
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    log_fn(f"Early stopping at epoch {epoch}; best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state["model_state_dict"])
            if checkpoint_path is not None:
                self.save_model(checkpoint_path, metadata=best_state)

        if history_path is not None:
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with history_path.open("w", encoding="utf-8") as handle:
                import csv

                writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

        return history

    @torch.no_grad()
    def evaluate(self, loader, criterion=None, threshold: float = 0.5) -> dict[str, Any]:
        self.model.eval()
        criterion = criterion or nn.BCEWithLogitsLoss()
        total_loss = 0.0
        y_true: list[int] = []
        y_prob: list[float] = []

        for images, targets, _meta in loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * images.size(0)
            y_true.extend(targets.detach().cpu().numpy().astype(int).tolist())
            y_prob.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())

        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_prob_arr = np.asarray(y_prob, dtype=np.float32)
        metrics = _binary_metrics(y_true_arr, y_prob_arr, threshold=threshold)
        metrics["loss"] = total_loss / max(1, len(loader.dataset))
        metrics["classification_report"] = classification_report(
            y_true_arr,
            (y_prob_arr >= threshold).astype(np.int64),
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        )
        metrics["y_true"] = y_true_arr.tolist()
        metrics["y_prob"] = y_prob_arr.tolist()
        return metrics

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, threshold: float = 0.5) -> dict[str, Any]:
        self.model.eval()
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        logits = self.model(image_tensor.to(self.device))
        prob = torch.sigmoid(logits).squeeze(0).item()
        pred = 1 if prob >= threshold else 0
        return {
            "label": CLASS_NAMES[pred],
            "confidence": prob if pred == 1 else 1.0 - prob,
            "equiaxed_probability": prob,
            "columnar_probability": 1.0 - prob,
        }

    def save_model(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata.copy() if metadata else {}
        payload.setdefault("class_names", CLASS_NAMES)
        payload["model_state_dict"] = self.model.state_dict()
        torch.save(payload, path)

    @classmethod
    def load_model(cls, path: str | Path, map_location: str | torch.device | None = None) -> "GrainCNNExperiment":
        checkpoint = torch.load(path, map_location=map_location or "cpu")
        experiment = cls()
        experiment.model.load_state_dict(checkpoint["model_state_dict"])
        return experiment
