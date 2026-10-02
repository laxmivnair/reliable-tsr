"""
Step 5 + orchestration: run TATR on a folder of images, compare each
prediction against its GT annotation, and save every table that has at
least one error into a single Real-Error Dataset JSON file.

Usage (on your own machine, with network access and the datasets downloaded):

    python build_error_dataset.py \
        --images_dir /path/to/images \
        --annotations_dir /path/to/annotations \
        --dataset pubtables1m \
        --out real_error_dataset.json \
        --limit 300

--dataset selects which loader in loaders.py to use (pubtables1m / fintabnet / scitsr).
--limit caps how many images to process (useful for the ~300-image diagnostic
sample discussed earlier -- stratify the file list yourself before passing it in
if you want controlled coverage of table complexity / source quality / domain).
"""

import argparse
import json
import os
from PIL import Image

from loaders import LOADERS
from error_extraction import extract_errors


def find_pairs(images_dir: str, annotations_dir: str, ann_ext: str):
    """Pairs each image with its annotation file by matching basenames."""
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        base, ext = os.path.splitext(fname)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        ann_path = os.path.join(annotations_dir, base + ann_ext)
        if os.path.exists(ann_path):
            pairs.append((os.path.join(images_dir, fname), ann_path, base))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--annotations_dir", required=True)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--out", default="real_error_dataset.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--score_threshold", type=float, default=0.6)
    ap.add_argument("--save_all", action="store_true",
                     help="Save every table, not just ones with errors "
                          "(useful if you also want clean examples for calibration).")
    args = ap.parse_args()

    ann_ext = ".xml" if args.dataset == "pubtables1m" else ".json"
    pairs = find_pairs(args.images_dir, args.annotations_dir, ann_ext)
    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Found {len(pairs)} image/annotation pairs for dataset={args.dataset}")

    # TATR import is deferred to here so the rest of the file can be unit
    # tested without torch/transformers installed.
    from tatr_inference import TATRPredictor
    predictor = TATRPredictor()

    gt_loader = LOADERS[args.dataset]
    entries = []

    for img_path, ann_path, base in pairs:
        image = Image.open(img_path).convert("RGB")
        gt_table = gt_loader(ann_path, image_id=base)
        pred_table = predictor.predict_table(image, image_id=base,
                                              score_threshold=args.score_threshold)

        result = extract_errors(gt_table, pred_table)
        if result["has_errors"] or args.save_all:
            entries.append(result)

        print(f"  {base}: {result['error_count']} errors")

    with open(args.out, "w") as f:
        json.dump(entries, f, indent=2)

    n_with_errors = sum(1 for e in entries if e["has_errors"])
    print(f"\nSaved {len(entries)} entries ({n_with_errors} with errors) to {args.out}")


if __name__ == "__main__":
    main()