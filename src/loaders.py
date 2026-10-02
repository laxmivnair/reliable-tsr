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
import os
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
    # encoding="utf-8" is required, not optional: without it, open() falls back
    # to the OS locale encoding -- cp1252 on Windows -- which crashes on any
    # non-ASCII byte in the file (UnicodeDecodeError). JSON is UTF-8 by spec.
    with open(json_path, encoding="utf-8") as f:
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
# SciTSR: structure/[ID].json gives row/col spans + text CONTENT per cell,
# but NO bounding box. Bounding boxes live separately in chunk/[ID].chunk,
# which is a list of OCR/PDF text chunks, each with its own "pos" in
# [x1, x2, y1, y2] order -- but chunks are NOT linked to cell IDs anywhere
# in the released files. We recover each cell's bbox by matching its text
# content against chunk text (see _align_chunks_to_cells below).
#
# Genuine limitation of the dataset, not a bug here: a cell with EMPTY
# content (a blank table cell) has no corresponding chunk and therefore no
# recoverable bbox. Such cells are dropped from Table.cells (counted in
# Table.skipped_no_bbox_count) since bbox/IoU matching cannot use them.
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    return " ".join(s.split()).strip().lower()


def _to_xyxy(chunk_pos):
    """Converts SciTSR's chunk pos [x1, x2, y1, y2] into [xmin, ymin, xmax, ymax]."""
    x1, x2, y1, y2 = chunk_pos
    return [x1, y1, x2, y2]


def _union_bbox(chunk_positions):
    """Union of several chunk positions (each in SciTSR's [x1,x2,y1,y2] order),
    returned in standard [xmin, ymin, xmax, ymax] order."""
    converted = [_to_xyxy(p) for p in chunk_positions]
    xs1 = [b[0] for b in converted]; ys1 = [b[1] for b in converted]
    xs2 = [b[2] for b in converted]; ys2 = [b[3] for b in converted]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def _align_chunks_to_cells(cells_raw, chunks):
    """
    Returns a dict: cell index in cells_raw -> bbox (xmin,ymin,xmax,ymax),
    for every cell that could be matched to one or more chunks.

    Strategy (greedy, good enough for the vast majority of cells, verify on
    your own sample before trusting it on anything load-bearing):
      1. Exact match: a cell's full joined content equals one chunk's text
         (handles the common case where one chunk == one cell).
      2. Token-level fallback: a cell's content is a list of words: Collect
         every remaining, unused chunk whose text matches one of those
         words exactly, and union their positions. Handles cells whose
         text was split across multiple chunks.
      Chunks are consumed once used, so one chunk cannot be double-assigned
      to two different cells.
    """
    chunk_text_to_indices = {}
    for i, ch in enumerate(chunks):
        chunk_text_to_indices.setdefault(_normalize(ch["text"]), []).append(i)

    used = set()
    cell_bbox = {}

    # Pass 1: exact full-text match
    for ci, c in enumerate(cells_raw):
        content = c.get("content", [])
        full_text = _normalize(" ".join(content)) if content else ""
        if not full_text:
            continue
        candidates = [i for i in chunk_text_to_indices.get(full_text, []) if i not in used]
        if candidates:
            idx = candidates[0]
            used.add(idx)
            cell_bbox[ci] = _to_xyxy(chunks[idx]["pos"])

    # Pass 2: token-level fallback for anything still unmatched
    for ci, c in enumerate(cells_raw):
        if ci in cell_bbox:
            continue
        content = c.get("content", [])
        if not content:
            continue
        matched_positions = []
        for word in content:
            norm_word = _normalize(word)
            candidates = [i for i in chunk_text_to_indices.get(norm_word, []) if i not in used]
            if candidates:
                idx = candidates[0]
                used.add(idx)
                matched_positions.append(chunks[idx]["pos"])
        if matched_positions:
            cell_bbox[ci] = _union_bbox(matched_positions)

    return cell_bbox


def load_scitsr(structure_json_path: str, image_id: str, chunk_path: str = None) -> Table:
    """
    structure_json_path: path to structure/[ID].json
    chunk_path: path to chunk/[ID].chunk. If omitted, we look for a sibling
    file next to structure_json_path with the same basename and a .chunk
    extension (this is what download_scitsr_sample.py produces).
    """
    # encoding="utf-8" required -- see note in load_fintabnet above. SciTSR's
    # chunk text comes from PDF extraction and commonly contains bytes (e.g.
    # ligatures, accented characters, curly quotes) that are flatly invalid
    # under Windows' default cp1252 locale encoding.
    with open(structure_json_path, encoding="utf-8") as f:
        data = json.load(f)
    cells_raw = data.get("cells", [])

    if chunk_path is None:
        base = os.path.splitext(structure_json_path)[0]
        chunk_path = base + ".chunk"
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(
            f"No chunk file found at {chunk_path}. SciTSR cell bboxes come from "
            f"chunk/[ID].chunk, not structure/[ID].json -- make sure both files "
            f"are present (download_scitsr_sample.py copies both)."
        )
    with open(chunk_path, encoding="utf-8") as f:
        chunk_data = json.load(f)
    chunks = chunk_data.get("chunks", [])

    cell_bbox = _align_chunks_to_cells(cells_raw, chunks)

    cells = []
    skipped = 0
    for ci, c in enumerate(cells_raw):
        bbox = cell_bbox.get(ci)
        if bbox is None:
            skipped += 1   # genuinely empty cell, or an alignment miss -- see docstring
            continue
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

    return Table(image_id=image_id, dataset_source="SciTSR", cells=cells,
                 skipped_no_bbox_count=skipped)


LOADERS = {
    "pubtables1m": load_pubtables1m,
    "fintabnet": load_fintabnet,
    "scitsr": load_scitsr,
}