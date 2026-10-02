"""
Step 1 (loader half): standardize GT annotation formats into schema.Table.

IMPORTANT: I wrote these against the documented/standard annotation layout
for each dataset, but I have not run them against your actual downloaded
files (no network access to PubTables-1M / FinTabNet / SciTSR in this
environment). Before trusting output in bulk, run each loader on 2-3 known
files and print the result next to the original annotation to confirm the
field names match what you actually have on disk -- annotation format has
drifted slightly across dataset releases.

Each loader takes a path to one annotation file (or XML tree) for one
table image and returns a schema.Table.
"""

import json
import xml.etree.ElementTree as ET
from typing import List
from schema import Cell, Table


# ---------------------------------------------------------------------------
# PubTables-1M: PASCAL VOC-style XML, one object per row/column/spanning
# cell/header, defined by axis-aligned boxes. Cells themselves are NOT
# given directly -- they are the grid intersections of row x column boxes,
# adjusted for spanning cells. This loader reconstructs cells from that grid.
# ---------------------------------------------------------------------------

def load_pubtables1m(xml_path: str, image_id: str) -> Table:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text) if size is not None else None
    img_h = float(size.find("height").text) if size is not None else None

    rows, cols, spanning_cells, headers = [], [], [], []

    for obj in root.findall("object"):
        label = obj.find("name").text
        bnd = obj.find("bndbox")
        box = [float(bnd.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
        if label == "table row":
            rows.append(box)
        elif label == "table column":
            cols.append(box)
        elif label == "table spanning cell":
            spanning_cells.append(box)
        elif label == "table column header":
            headers.append(box)

    rows.sort(key=lambda b: b[1])   # top to bottom
    cols.sort(key=lambda b: b[0])   # left to right

    def row_overlaps(box, row_box):
        return not (box[3] <= row_box[1] or box[1] >= row_box[3])

    def col_overlaps(box, col_box):
        return not (box[2] <= col_box[0] or box[0] >= col_box[2])

    header_row_idxs = {ri for ri, r in enumerate(rows)
                        for h in headers if row_overlaps(h, r)}

    cells = []
    # Base grid: one cell per (row, col) intersection
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            bbox = [c[0], r[1], c[2], r[3]]
            cells.append(Cell(
                bbox=bbox, start_row=ri, end_row=ri, start_col=ci, end_col=ci,
                is_header=(ri in header_row_idxs),
            ))

    # Spanning cells override: find which grid cells a spanning box covers,
    # remove them, replace with one merged cell.
    for sc in spanning_cells:
        covered_rows = [ri for ri, r in enumerate(rows) if row_overlaps(sc, r)]
        covered_cols = [ci for ci, c in enumerate(cols) if col_overlaps(sc, c)]
        if not covered_rows or not covered_cols:
            continue
        cells = [c for c in cells if not (
            c.start_row in covered_rows and c.start_col in covered_cols)]
        cells.append(Cell(
            bbox=sc, start_row=min(covered_rows), end_row=max(covered_rows),
            start_col=min(covered_cols), end_col=max(covered_cols),
            is_header=any(r in header_row_idxs for r in covered_rows),
        ))

    return Table(image_id=image_id, dataset_source="PubTables-1M",
                 cells=cells, image_width=img_w, image_height=img_h)


# ---------------------------------------------------------------------------
# FinTabNet: released as per-table JSON ("cells" list with bbox + row/col
# span + text), same general family of annotation as PubTables-1M since
# both come from the Smock et al. "Aligning benchmark datasets" releases.
# ---------------------------------------------------------------------------

def load_fintabnet(json_path: str, image_id: str) -> Table:
    with open(json_path) as f:
        data = json.load(f)

    cells = []
    for c in data.get("cells", []):
        cells.append(Cell(
            bbox=list(c["bbox"]),
            start_row=int(c["start_row"]),
            end_row=int(c["end_row"]),
            start_col=int(c["start_col"]),
            end_col=int(c["end_col"]),
            text_content=c.get("text", ""),
            is_header=bool(c.get("is_header", False)),
        ))

    return Table(
        image_id=image_id, dataset_source="FinTabNet", cells=cells,
        image_width=data.get("width"), image_height=data.get("height"),
    )


# ---------------------------------------------------------------------------
# SciTSR: JSON with a "cells" list; each cell gives "content" (tokens),
# "start_row"/"end_row"/"start_col"/"end_col", and a "pos" bbox in
# [x1, x2, y1, y2] order (note: x-pair then y-pair, not the usual
# xmin,ymin,xmax,ymax -- this is the actual SciTSR convention).
# ---------------------------------------------------------------------------

def load_scitsr(json_path: str, image_id: str) -> Table:
    with open(json_path) as f:
        data = json.load(f)

    cells = []
    for c in data.get("cells", []):
        x1, x2, y1, y2 = c["pos"]
        bbox = [x1, y1, x2, y2]
        text = " ".join(c.get("content", [])) if isinstance(c.get("content"), list) \
            else c.get("content", "")
        cells.append(Cell(
            bbox=bbox,
            start_row=int(c["start_row"]),
            end_row=int(c["end_row"]),
            start_col=int(c["start_col"]),
            end_col=int(c["end_col"]),
            text_content=text,
            is_header=bool(c.get("is_header", False)),
        ))

    return Table(image_id=image_id, dataset_source="SciTSR", cells=cells)


LOADERS = {
    "pubtables1m": load_pubtables1m,
    "fintabnet": load_fintabnet,
    "scitsr": load_scitsr,
}