# Phong Tran
# This program runs the models and prints out the expected results
import os
import sys
import math
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet18

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 224          
SCORE_THRESH = 0.45     
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----------------------------
# Preprocessing helpers
# ----------------------------
def center_square_crop_and_resize_no_balls(img, out_size=224):
    """
    Same crop logic as training, but without needing annotations.
    Returns resized_image, (off_x, off_y, siz, scale).
    """
    h, w = img.shape[:2]
    siz = min(h, w)
    off_x = (w - siz) // 2
    off_y = (h - siz) // 2
    crop = img[off_y:off_y + siz, off_x:off_x + siz]
    resized = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    scale = out_size / float(siz)
    return resized, (off_x, off_y, siz, scale)

# ----------------------------
# Model loading
# ----------------------------
def load_detector(weights_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    num_classes = 3  # background, solid, striped (trained that way)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location="cpu")
    # convert half -> float32
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            state_dict[k] = v.float()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_classifier(weights_path, device):
    # same architecture as training: ResNet18 with 2 outputs
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)

    state_dict = torch.load(weights_path, map_location="cpu")
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            state_dict[k] = v.float()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ----------------------------
# Patch extraction for classifier
# ----------------------------
def extract_ball_patch(img_bgr, x, y, r, out_size=224, scale=1.3):
    """
    Crop a square patch around (x,y) with radius r, scaled by 'scale'.
    Returns normalized tensor (1,3,H,W).
    """
    h, w = img_bgr.shape[:2]
    half = r * scale
    x1 = int(max(0, x - half))
    y1 = int(max(0, y - half))
    x2 = int(min(w - 1, x + half))
    y2 = int(min(h - 1, y + half))

    if x2 <= x1 or y2 <= y1:
        # degenerate, fallback to small patch
        x1, y1 = max(0, int(x) - 5), max(0, int(y) - 5)
        x2, y2 = min(w - 1, int(x) + 5), min(h - 1, int(y) + 5)

    patch = img_bgr[y1:y2, x1:x2]
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

    patch = patch.astype(np.float32) / 255.0
    patch = (patch - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return patch

# ----------------------------
# Detection + classification
# ----------------------------
def detect_balls(detector, classifier, img_bgr, device):
    """
    Returns list of detections [(X,Y,R,V), ...] in original image coords.
    V = 0 solid, 1 striped (from classifier with confidence heuristic).
    """
    orig_h, orig_w = img_bgr.shape[:2]

    # center crop + resize as during training
    cropped, (off_x, off_y, siz, scale) = center_square_crop_and_resize_no_balls(
        img_bgr, out_size=IMG_SIZE
    )
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    to_tensor = torchvision.transforms.ToTensor()
    tensor_img = to_tensor(img_rgb).to(device)

    with torch.no_grad():
        outputs = detector([tensor_img])[0]

    boxes = outputs.get("boxes", torch.empty((0, 4))).cpu().numpy()
    scores = outputs.get("scores", torch.empty((0,))).cpu().numpy()
    labels = outputs.get("labels", torch.empty((0,), dtype=torch.long)).cpu().numpy()

    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score < SCORE_THRESH:
            continue
        if label == 0:
            continue  # background, ignore

        xmin, ymin, xmax, ymax = box
        xmin = float(np.clip(xmin, 0, IMG_SIZE - 1))
        ymin = float(np.clip(ymin, 0, IMG_SIZE - 1))
        xmax = float(np.clip(xmax, 0, IMG_SIZE - 1))
        ymax = float(np.clip(ymax, 0, IMG_SIZE - 1))
        w_res = xmax - xmin
        h_res = ymax - ymin
        if w_res <= 0 or h_res <= 0:
            continue

        cx_res = 0.5 * (xmin + xmax)
        cy_res = 0.5 * (ymin + ymax)
        r_res = 0.5 * min(w_res, h_res)

        if scale <= 0:
            continue

        # map back to original coords
        cx_orig = cx_res / scale + off_x
        cy_orig = cy_res / scale + off_y
        r_orig = r_res / scale

        X = int(round(cx_orig))
        Y = int(round(cy_orig))
        R = int(round(r_orig))

        X = max(0, min(orig_w - 1, X))
        Y = max(0, min(orig_h - 1, Y))

        # radius sanity check (helps FAR)
        min_r = 3
        max_r = max(4, min(orig_w, orig_h) // 8)  # balls shouldn't be huge
        if R < min_r or R > max_r:
            continue

        if R <= 0:
            continue

        results.append((X, Y, R, float(score)))

    # sort by score (desc) and basic NMS-like filtering on centers
    results.sort(key=lambda t: t[3], reverse=True)
    kept = []
    used = []
    for x, y, r, s in results:
        keep = True
        for ux, uy, ur, us in used:
            dist = math.hypot(x - ux, y - uy)
            if dist < 0.5 * (r + ur):
                keep = False
                break
        if keep:
            used.append((x, y, r, s))
            kept.append((x, y, r))

    # classify each kept detection with ResNet18 (solid vs striped), with a confidence heuristic: if not confident, bias to solid (0).
    final = []
    with torch.no_grad():
        for (x, y, r) in kept:
            patch = extract_ball_patch(img_bgr, x, y, r, out_size=224, scale=1.3)
            patch = patch.to(device)
            logits = classifier(patch)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            cls = int(probs.argmax())
            conf = float(probs[cls])

            # Tiny heuristic: if classifier is uncertain, default to solid (0)
            if conf < 0.6:
                cls = 0

            final.append((x, y, r, cls))

    return final

# ----------------------------
# Main / CLI
# ----------------------------
def main():
    if len(sys.argv) != 2:
        sys.stderr.write("Usage: python project3.py /path/to/image.png\n")
        sys.exit(1)

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        sys.stderr.write(f"Error: image file not found: {img_path}\n")
        sys.exit(1)

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        sys.stderr.write(f"Error: could not read image: {img_path}\n")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    det_weights = os.path.join(script_dir, "balldetector_frcnn.pth")
    cls_weights = os.path.join(script_dir, "balltype_resnet18.pth")

    if not os.path.exists(det_weights):
        sys.stderr.write(f"Error: detector weights not found: {det_weights}\n")
        sys.exit(1)
    if not os.path.exists(cls_weights):
        sys.stderr.write(f"Error: classifier weights not found: {cls_weights}\n")
        sys.exit(1)

    detector = load_detector(det_weights, device)
    classifier = load_classifier(cls_weights, device)

    detections = detect_balls(detector, classifier, img_bgr, device)

    # Output: N, then N lines "X Y R V"
    print(len(detections))
    for (x, y, r, v) in detections:
        print(f"{x} {y} {r} {v}")

if __name__ == "__main__":
    main()
