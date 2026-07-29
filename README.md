# GrainStructureAnalyzingModel

Custom grain-morphology CNN project for classifying columnar vs. equiaxed grain patches.

## Current workflow

The scratch-CNN pipeline lives in the repository root:

- [grain_cnn_model.py](grain_cnn_model.py) - custom CNN architecture and training/evaluation wrapper
- [grain_data_loader.py](grain_data_loader.py) - source-aware dataset loading and augmentation
- [train_grain_cnn.py](train_grain_cnn.py) - trains the custom CNN from scratch
- [evaluate_grain_cnn.py](evaluate_grain_cnn.py) - test metrics, confusion matrix, example predictions, and Dr. Lu source summary
- [compare_architectures.py](compare_architectures.py) - side-by-side parameter and metric comparison against the older EfficientNet run

## Dataset layout

The dataset in this repo is already grouped by source image and class under [Dataset/grain_crops](Dataset/grain_crops), with the manifests generated in [artifacts/splits](artifacts/splits).

Train the custom model from the repo root like this:

```bash
./.venv/bin/python train_grain_cnn.py --dataset-root Dataset --split-dir artifacts/splits --output-dir artifacts/grain_cnn_run --epochs 60 --batch-size 32
```

Then evaluate it:

```bash
./.venv/bin/python evaluate_grain_cnn.py --dataset-root Dataset --split-dir artifacts/splits --checkpoint artifacts/grain_cnn_run/best_model.pt --output-dir artifacts/grain_cnn_run/eval
```
