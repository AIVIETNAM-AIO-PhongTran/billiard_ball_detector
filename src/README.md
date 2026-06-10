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
├── inference.py
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
python src/inference.py assets/sample_input.png
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
15
306 104 22 1
222 230 21 1
224 120 20 0
264 113 20 0
230 191 20 0
328 199 21 1
276 185 21 1
185 287 21 1
174 172 21 0
290 145 21 1
149 244 21 0
184 134 19 1
142 136 24 0
248 153 19 0
249 260 21 1
```
