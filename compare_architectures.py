from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from grain_cnn_model import GrainCNNExperiment, build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the scratch CNN against the earlier EfficientNet-B0 run.")
    parser.add_argument("--custom-checkpoint", type=str, default="artifacts/grain_cnn_run/best_model.pt")
    parser.add_argument("--custom-metrics", type=str, default="artifacts/grain_cnn_run/metrics.json")
    parser.add_argument("--baseline-checkpoint", type=str, default="artifacts/model_run/best_model.pt")
    parser.add_argument("--baseline-metrics", type=str, default="artifacts/model_run/metrics.json")
    return parser.parse_args()


def count_effnet_params() -> dict[str, int]:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 2)
    return count_parameters(model)


def load_metric(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    custom_experiment = GrainCNNExperiment.load_model(args.custom_checkpoint) if Path(args.custom_checkpoint).exists() else GrainCNNExperiment(build_model())

    custom_params = count_parameters(custom_experiment.model)
    effnet_params = count_effnet_params()

    custom_metrics = load_metric(args.custom_metrics)
    baseline_metrics = load_metric(args.baseline_metrics)

    print("Architecture comparison")
    print(f"Custom CNN params: {custom_params['total']:,} total / {custom_params['trainable']:,} trainable")
    print(f"EfficientNet-B0 params: {effnet_params['total']:,} total / {effnet_params['trainable']:,} trainable")
    print("")
    print("Metrics")
    if custom_metrics:
        custom_val = custom_metrics.get('validation', {})
        custom_test = custom_metrics.get('test', {})
        print(
            f"Custom CNN val accuracy: {custom_val.get('accuracy', 'n/a')} | "
            f"val loss: {custom_val.get('loss', 'n/a')} | "
            f"test accuracy: {custom_test.get('accuracy', 'n/a')}"
        )
    else:
        print("Custom CNN metrics not found yet.")
    if baseline_metrics:
        baseline_val = baseline_metrics.get('validation', {})
        print(
            f"EfficientNet-B0 val accuracy: {baseline_val.get('accuracy', 'n/a')} | "
            f"val loss: {baseline_val.get('loss', 'n/a')} | "
            f"test accuracy: {baseline_metrics.get('test', {}).get('accuracy', 'n/a')}"
        )
    else:
        print("EfficientNet-B0 metrics not found yet.")


if __name__ == "__main__":
    main()