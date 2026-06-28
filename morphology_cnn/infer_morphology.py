from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

IMAGE_SIZE = 256
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(num_classes: int = 2):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer grain morphology and a qualitative material-property note.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt from training.")
    parser.add_argument("--image", type=str, required=True, help="Path to an image to classify.")
    parser.add_argument(
        "--equivalent-diameter-px",
        type=float,
        default=None,
        help="Optional grain-size proxy in pixels for qualitative Hall-Petch commentary.",
    )
    parser.add_argument(
        "--comparison-equivalent-diameter-px",
        type=float,
        default=None,
        help="Optional comparison grain size in pixels from another region of the same alloy/scale.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    return parser.parse_args()


def classify(image_path: Path, checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", ["columnar", "equiaxed"])

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = build_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), {name: float(probs[i]) for i, name in enumerate(class_names)}


def property_inference(label: str, confidence: float, equivalent_diameter_px: float | None, comparison_equivalent_diameter_px: float | None) -> list[str]:
    notes = []
    notes.append("This dataset has no paired mechanical-property labels, so no trained regression to yield strength/hardness is being claimed here.")

    if label == "columnar":
        notes.append(
            "Columnar-dominant morphology is usually consistent with directional solidification or a strong thermal gradient, so anisotropic properties are plausible. This is a qualitative metallurgical relationship, not a numerical prediction."
        )
    else:
        notes.append(
            "Equiaxed morphology is usually consistent with more isotropic grain shapes and often less directionally solidified structure. This is a qualitative metallurgical relationship, not a numerical prediction."
        )

    if equivalent_diameter_px is None and comparison_equivalent_diameter_px is None:
        notes.append(
            "No grain-size proxy was supplied, so Hall-Petch-style comparison is not applied. From morphology alone, a mechanical-property number cannot be determined honestly."
        )
    else:
        if equivalent_diameter_px is not None and comparison_equivalent_diameter_px is not None:
            if equivalent_diameter_px < comparison_equivalent_diameter_px:
                notes.append(
                    "Hall-Petch comparative note: the supplied region has a smaller equivalent diameter than the comparison region, so it would generally be expected to have relatively higher yield strength than the coarser-grained comparison region of the same alloy at the same scale. No absolute MPa value is being inferred."
                )
            elif equivalent_diameter_px > comparison_equivalent_diameter_px:
                notes.append(
                    "Hall-Petch comparative note: the supplied region has a larger equivalent diameter than the comparison region, so it would generally be expected to have relatively lower yield strength than the finer-grained comparison region of the same alloy at the same scale. No absolute MPa value is being inferred."
                )
            else:
                notes.append(
                    "Hall-Petch comparative note: the supplied grain-size proxy matches the comparison region, so no relative strength advantage can be stated from size alone. No absolute MPa value is being inferred."
                )
        else:
            notes.append(
                "A grain-size proxy was supplied, but there is no comparison region. Hall-Petch can only be used here in a comparative way, so the correct statement is that smaller grains would generally suggest relatively higher yield strength than a coarser region of the same alloy. No absolute MPa value is being inferred."
            )

    if confidence < 0.7:
        notes.append(
            "Prediction confidence is modest, so the property note should be treated as a weak qualitative hint only."
        )

    return notes


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    label, confidence, probs = classify(image_path, checkpoint_path)
    notes = property_inference(
        label=label,
        confidence=confidence,
        equivalent_diameter_px=args.equivalent_diameter_px,
        comparison_equivalent_diameter_px=args.comparison_equivalent_diameter_px,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "classification_result": {
                        "label": label,
                        "confidence": confidence,
                        "class_probabilities": probs,
                    },
                    "property_inference": notes,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    print("Classification result")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {confidence:.4f}")
    print("  Class probabilities:")
    for name, prob in sorted(probs.items(), key=lambda kv: kv[0]):
        print(f"    - {name}: {prob:.4f}")
    print()
    print("Property inference")
    for note in notes:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
