"""
Step 4: Rule-based error extraction & categorization.

Takes a GT Table and a predicted Table (TATR output, already converted to
the unified schema) and returns a structured dict of every error found,
in the same shape used in Step 5 of the spec.
"""

from typing import Dict, Any, List
from schema import Table
from matching import match_cells, MatchResult

# Separate thresholds for two different jobs:
#  - MATCH_IOU_THRESHOLD: "is this predicted cell basically the same cell as this GT cell"
#    (used for the strict one-to-one match -> missing/extra/span errors)
#  - OVERLAP_IOU_THRESHOLD: "is this predicted cell touching this GT cell at all"
#    (used for merge/split detection, which is inherently many-to-one)
MATCH_IOU_THRESHOLD = 0.5
OVERLAP_IOU_THRESHOLD = 0.1


def _bbox_list(bbox) -> List[float]:
    return [round(float(x), 2) for x in bbox]


def find_merges_and_splits(gt: Table, pred: Table, result: MatchResult) -> Dict[str, List[dict]]:
    """
    Incorrect merge: one predicted cell overlaps >1 distinct GT cells.
    Incorrect split:  one GT cell overlaps >1 distinct predicted cells.
    """
    n_gt, n_pred = len(gt.cells), len(pred.cells)
    merges, splits = [], []

    # predicted cell -> list of GT indices it overlaps
    for j in range(n_pred):
        overlapping_gt = [i for i in range(n_gt)
                           if result.iou_matrix[i][j] >= OVERLAP_IOU_THRESHOLD]
        if len(overlapping_gt) > 1:
            merges.append({
                "predicted_bbox": _bbox_list(pred.cells[j].bbox),
                "merged_gt_bboxes": [_bbox_list(gt.cells[i].bbox) for i in overlapping_gt],
                "merged_gt_spans": [
                    {"row": [gt.cells[i].start_row, gt.cells[i].end_row],
                     "col": [gt.cells[i].start_col, gt.cells[i].end_col]}
                    for i in overlapping_gt
                ],
            })

    # GT cell -> list of predicted indices overlapping it
    for i in range(n_gt):
        overlapping_pred = [j for j in range(n_pred)
                             if result.iou_matrix[i][j] >= OVERLAP_IOU_THRESHOLD]
        if len(overlapping_pred) > 1:
            splits.append({
                "gt_bbox": _bbox_list(gt.cells[i].bbox),
                "gt_span": {"row": [gt.cells[i].start_row, gt.cells[i].end_row],
                            "col": [gt.cells[i].start_col, gt.cells[i].end_col]},
                "split_into_predicted_bboxes": [_bbox_list(pred.cells[j].bbox) for j in overlapping_pred],
            })

    return {"incorrect_merges": merges, "incorrect_splits": splits}


def find_span_errors(gt: Table, pred: Table, result: MatchResult) -> List[dict]:
    """
    For every strictly-matched (GT, pred) pair, compare row/col span.
    A mismatch in start/end row OR start/end col is a span error.
    """
    span_errors = []
    for gt_idx, pred_idx, score in result.matched_pairs:
        g, p = gt.cells[gt_idx], pred.cells[pred_idx]
        row_mismatch = (g.start_row != p.start_row) or (g.end_row != p.end_row)
        col_mismatch = (g.start_col != p.start_col) or (g.end_col != p.end_col)
        if row_mismatch or col_mismatch:
            span_errors.append({
                "bbox": _bbox_list(g.bbox),
                "match_iou": round(score, 3),
                "predicted_span": {"row": [p.start_row, p.end_row], "col": [p.start_col, p.end_col]},
                "true_span": {"row": [g.start_row, g.end_row], "col": [g.start_col, g.end_col]},
                "row_span_error": row_mismatch,
                "col_span_error": col_mismatch,
            })
    return span_errors


def flag_header_errors(gt: Table, span_errors: List[dict], merges: List[dict]) -> List[dict]:
    """
    D. Any span error or merge whose GT row falls in a GT header row gets
    tagged header_alignment_error = True. Returns the same records, with
    the tag added, filtered down to the ones that are actually header errors.
    """
    header_rows = gt.header_row_indices()
    header_errors = []

    for e in span_errors:
        gt_rows = range(e["true_span"]["row"][0], e["true_span"]["row"][1] + 1)
        if any(r in header_rows for r in gt_rows):
            tagged = dict(e)
            tagged["header_alignment_error"] = True
            tagged["error_type"] = "span_error"
            header_errors.append(tagged)

    for m in merges:
        gt_rows = set()
        for s in m["merged_gt_spans"]:
            gt_rows.update(range(s["row"][0], s["row"][1] + 1))
        if any(r in header_rows for r in gt_rows):
            tagged = dict(m)
            tagged["header_alignment_error"] = True
            tagged["error_type"] = "incorrect_merge"
            header_errors.append(tagged)

    return header_errors


def extract_errors(gt: Table, pred: Table) -> Dict[str, Any]:
    """Main entry point: run the full Step 3 + Step 4 pipeline for one table."""
    result = match_cells(gt.cells, pred.cells, iou_threshold=MATCH_IOU_THRESHOLD)

    missing_cells = [_bbox_list(gt.cells[i].bbox) for i in result.unmatched_gt_idx]
    extra_cells = [_bbox_list(pred.cells[j].bbox) for j in result.unmatched_pred_idx]

    merge_split = find_merges_and_splits(gt, pred, result)
    span_errors = find_span_errors(gt, pred, result)
    header_errors = flag_header_errors(gt, span_errors, merge_split["incorrect_merges"])

    errors_found = {
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "incorrect_merges": merge_split["incorrect_merges"],
        "incorrect_splits": merge_split["incorrect_splits"],
        "row_col_relationship_errors": span_errors,
        "header_alignment_errors": header_errors,
    }

    n_errors = (len(missing_cells) + len(extra_cells)
                + len(merge_split["incorrect_merges"]) + len(merge_split["incorrect_splits"])
                + len(span_errors))

    return {
        "image_id": gt.image_id,
        "dataset_source": gt.dataset_source,
        "has_errors": n_errors > 0,
        "error_count": n_errors,
        "errors_found": errors_found,
    }