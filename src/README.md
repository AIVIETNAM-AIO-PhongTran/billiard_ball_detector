# Source Code

This directory contains the inference implementation used by the Billiard Ball Detector project.

---

## Contents

```text
src/
├── README.md
└── inference.py
```

---

## inference.py

Main inference script.

Responsibilities:

1. Load trained Faster R-CNN detector
2. Load trained ResNet18 classifier
3. Read input image
4. Detect billiard balls
5. Classify each detected ball
6. Produce formatted output

---

## Required Model Files

The script expects the following trained model weights to be located in the same directory:

```text
src/
├── project3.py
├── balldetector_frcnn.pth
└── balltype_resnet18.pth
```

These files are generated after training completes in:

```text
model_train.ipynb
```
Since the files are generated and downloaded from Google Collab, please make sure to add them to src before running the inference script.
---

## Running

From the project root:

```bash
python src/inference.py path/to/image.png
```

Example:

```bash
python src/project3.py assets/sample_input.png
```

---

## Output Format

```text
N
X Y R V
```

Where:

| Variable | Meaning             |
| -------- | ------------------- |
| X        | Center x-coordinate |
| Y        | Center y-coordinate |
| R        | Estimated radius    |
| V        | Ball type           |

```text
0 = Solid
1 = Striped
```

Example:

```text
16
302 506 102 0
888 570 100 0
948 1240 117 0
425 659 107 0
523 491 103 0
1030 260 97 1
572 801 109 0
830 290 96 1
830 772 107 0
612 293 96 1
730 946 112 0
672 638 104 0
965 419 98 1
417 336 99 1
206 350 102 1
737 441 100 1
```
