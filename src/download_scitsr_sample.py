"""
Downloads SciTSR (hosted on Google Drive by the dataset authors) and copies
out a stratified sample of N images + their structure JSON into flat
images_dir / annotations_dir folders, ready for build_error_dataset.py.

Run this on your own machine (needs internet + ~2-5 GB free space for the
full SciTSR zip, even though we only use a subset of it).

Usage:
    pip install gdown
    python download_scitsr_sample.py --n 300 --out ./scitsr_sample
"""

import argparse
import os
import random
import shutil
import subprocess
import zipfile

SCITSR_GDRIVE_ID = "1qXaJblBg9sbPN0xknWsYls1aGGtlp4ZN"  # from the official SciTSR README


def download_and_extract(workdir: str):
    zip_path = os.path.join(workdir, "SciTSR.zip")
    if not os.path.exists(zip_path):
        print("Downloading SciTSR from Google Drive (this is a multi-GB file, "
              "may take a while)...")
        subprocess.run(["gdown", "--id", SCITSR_GDRIVE_ID, "-O", zip_path], check=True)
    extract_dir = os.path.join(workdir, "SciTSR")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(workdir)
    return extract_dir


def build_sample(scitsr_dir: str, split: str, n: int, out_dir: str, seed: int = 42):
    img_dir = os.path.join(scitsr_dir, split, "img")
    structure_dir = os.path.join(scitsr_dir, split, "structure")
    chunk_dir = os.path.join(scitsr_dir, split, "chunk")

    ids = [os.path.splitext(f)[0] for f in os.listdir(structure_dir) if f.endswith(".json")]
    random.Random(seed).shuffle(ids)
    chosen = ids[:n]

    images_out = os.path.join(out_dir, "images")
    annotations_out = os.path.join(out_dir, "annotations")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(annotations_out, exist_ok=True)

    copied, missing_chunk = 0, 0
    for table_id in chosen:
        chunk_src = os.path.join(chunk_dir, table_id + ".chunk")
        if not os.path.exists(chunk_src):
            # structure.json without a matching chunk file can't get bboxes at all --
            # skip it rather than copy a file load_scitsr() will just fail on later.
            missing_chunk += 1
            continue
        # SciTSR images may be .png or .jpg depending on release version
        for ext in (".png", ".jpg", ".jpeg"):
            src_img = os.path.join(img_dir, table_id + ext)
            if os.path.exists(src_img):
                shutil.copy(src_img, os.path.join(images_out, table_id + ext))
                shutil.copy(os.path.join(structure_dir, table_id + ".json"),
                            os.path.join(annotations_out, table_id + ".json"))
                # load_scitsr() looks for a sibling <id>.chunk next to <id>.json
                shutil.copy(chunk_src, os.path.join(annotations_out, table_id + ".chunk"))
                copied += 1
                break

    print(f"Copied {copied}/{len(chosen)} requested tables into {out_dir}")
    if missing_chunk:
        print(f"  (skipped {missing_chunk} with no .chunk file available)")
    print(f"  images:      {images_out}")
    print(f"  annotations: {annotations_out}  (each table: <id>.json + <id>.chunk)")
    print("\nNote: this is a RANDOM sample, not stratified by table complexity. "
          "SciTSR-COMP.list (in the zip root) lists IDs of complicated tables --"
          "mix some of those in deliberately if you want harder cases represented, "
          "e.g. by reading that list and forcing half of 'chosen' from it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--split", choices=["train", "test"], default="test")
    ap.add_argument("--out", default="./scitsr_sample")
    ap.add_argument("--workdir", default="./scitsr_download")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    scitsr_dir = download_and_extract(args.workdir)
    scitsr_dir = os.path.join(scitsr_dir, "SciTSR")
    build_sample(scitsr_dir, args.split, args.n, args.out)


if __name__ == "__main__":
    main()