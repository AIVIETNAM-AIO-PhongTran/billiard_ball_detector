# Dataset

This directory contains the annotated dataset used to train the detection and classification models for the Billiard Ball Detector project.

---

## Dataset Structure

Each image is paired with a corresponding annotation file.

```text
annotated_images/
├── 1.png
├── 1.txt
├── 2.png
├── 2.txt
├── 3.png
├── 3.txt
└── ...
```

For every image:

```text
N.png
```

there must be a matching annotation file:

```text
N.txt
```

where `N` is the same identifier.

Example:

```text
1.png
1.txt

15.png
15.txt
```

---

## Image Files

The `.png` files contain billiard table images used during training.

Each image may contain:

* Multiple billiard balls
* Different ball arrangements
* Different lighting conditions
* Different camera viewpoints
* Partial occlusions

---

## Annotation Files

The `.txt` files contain the ground-truth annotations associated with each image.

Annotations describe the billiard balls present in the corresponding image and are used to supervise model training.

Example relationship:

```text
1.png
└── Image containing billiard balls

1.txt
└── Ground-truth labels for 1.png
```

---

## Purpose

The dataset supports two machine learning tasks:

### Object Detection

Training the Faster R-CNN detector to:

* Locate billiard balls
* Learn ball boundaries
* Estimate object positions

### Classification

Training the ResNet18 classifier to distinguish between:

* Solid balls
* Striped balls

---

## Training Usage

The dataset is loaded in:

```text
model_train.ipynb
```

When training in Google Colab:

1. Upload the entire `annotated_images` directory to Google Drive.
2. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Set the dataset path:

```python
ROOT = "/content/drive/MyDrive/annotated_images"
```

The notebook expects both image files and annotation files to remain together in the same directory.

---

## Important Notes

* Do not rename image files without renaming their corresponding annotation files.
* Every image must have a matching `.txt` annotation file.
* Missing annotation files will cause training samples to be skipped or fail to load.
* The dataset structure should remain unchanged unless the notebook code is updated accordingly.

