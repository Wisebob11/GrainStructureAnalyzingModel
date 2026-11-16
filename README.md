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
./.venv/bin/python train_grain_cnn.py --dataset-root Dataset_merged --split-dir artifacts/splits --output-dir artifacts/grain_cnn_run --epochs 60 --batch-size 32
```

Then evaluate it:

```bash
./.venv/bin/python evaluate_grain_cnn.py --dataset-root Dataset_merged --split-dir artifacts/splits --checkpoint artifacts/grain_cnn_run/best_model.pt --output-dir artifacts/grain_cnn_run/eval
```

The split command requires explicit source choices:

```bash
./.venv/bin/python morphology_cnn/split_dataset.py --dataset-root Dataset_merged \
  --output-dir artifacts/splits --test-source Dr.Lu_Grain_Structure_Image \
  --val-source Area_1_1_IPF_A1.tiff
```

By default training uses a weighted sampler as the single class-imbalance
correction. Use `--imbalance-correction loss` or `none` for controlled
ablations, but do not combine sampler and weighted loss.

For the shape-only ablation, add `--input-mode shape`; this removes RGB/IPF
color information while retaining the same three-channel CNN input shape.
