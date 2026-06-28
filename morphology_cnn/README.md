# Grain morphology CNN

This folder contains a transfer-learning CNN pipeline for classifying grain morphology as columnar or equiaxed, plus a clearly separated rule-based property-inference layer.

## What this pipeline does

- Reads only the known class folders and image files under the dataset root.
- Uses the CSV metadata to create reproducible manifests.
- Splits by source image before training to avoid patch leakage.
- Trains a pretrained EfficientNet-B0 classifier with augmentation, weighted loss, and weighted oversampling.
- Reports accuracy, balanced accuracy, per-class precision/recall/F1, and a confusion matrix.
- Produces a qualitative property-inference note only; it does **not** train a regression model for MPa or hardness.

## Important scope note

The repository currently contains a per-grain segmentation dataset with two source images, not the smaller 81-patch dataset described in the prompt. That means the source-level split is effectively one source image for training/validation and one source image for testing. That is still the correct way to avoid leakage, but it is a very small and domain-shift-sensitive evaluation.

## Files

- [split_dataset.py](split_dataset.py): creates `train_manifest.csv`, `val_manifest.csv`, and `test_manifest.csv`
- [train_cnn.py](train_cnn.py): trains the classifier and writes metrics, a checkpoint, and a confusion matrix figure
- [infer_morphology.py](infer_morphology.py): predicts morphology and prints the rule-based property inference
- [common.py](common.py): shared dataset and file-handling helpers

## Typical workflow

1. Create manifests with the source-level split.
2. Train the classifier using the saved manifests.
3. Run inference on a new image.

## Property inference rules

- Columnar-dominant output -> qualitative note about directional solidification and anisotropy.
- Equiaxed output -> qualitative note about a more isotropic morphology.
- Optional grain-size proxy -> comparative Hall-Petch note only.
- No grain-size data -> no strength number is inferred.

## Example output structure

- Classification result
  - predicted class
  - confidence
  - class probabilities
- Property inference
  - short qualitative metallurgical note
  - explicit reminder that no absolute mechanical property is being predicted
