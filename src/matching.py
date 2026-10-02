"""
Step 3: Align predictions with ground truth.

Matches TATR's predicted cells against GT cells using IoU, and returns
three lists: matched pairs, unmatched predictions, unmatched GT cells.

Uses a greedy highest-IoU-first matching (not Hungarian/optimal assignment)
because table cells are spatially well-separated in almost all cases, so
greedy matching gives the same result as optimal assignment in practice
and is much simpler to reason about and debug. Each predicted cell and
each GT cell can be matched at most once.
"""

from dataclasses import dataclass
from typing import List, Tuple
from schema import Cell


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard axis-aligned IoU. Boxes are [xmin, ymin, xmax, ymax]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class MatchResult:
    matched_pairs: List[Tuple[int, int, float]]   # (gt_idx, pred_idx, iou)
    unmatched_pred_idx: List[int]
    unmatched_gt_idx: List[int]
    # raw IoU matrix kept around for the merge/split detection in error_extraction.py
    iou_matrix: List[List[float]]


def match_cells(gt_cells: List[Cell], pred_cells: List[Cell],
                 iou_threshold: float = 0.5) -> MatchResult:
    """
    Greedy, highest-IoU-first, one-to-one matching between GT and predicted
    cells. Any pair below iou_threshold is never matched.
    """
    n_gt, n_pred = len(gt_cells), len(pred_cells)
    iou_matrix = [[iou(gt_cells[i].bbox, pred_cells[j].bbox) for j in range(n_pred)]
                  for i in range(n_gt)]

    # Build a flat list of candidate (iou, gt_idx, pred_idx) above threshold, sorted desc.
    candidates = []
    for i in range(n_gt):
        for j in range(n_pred):
            if iou_matrix[i][j] >= iou_threshold:
                candidates.append((iou_matrix[i][j], i, j))
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt, matched_pred = set(), set()
    matched_pairs = []
    for score, i, j in candidates:
        if i in matched_gt or j in matched_pred:
            continue
        matched_gt.add(i)
        matched_pred.add(j)
        matched_pairs.append((i, j, score))

    unmatched_gt_idx = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred_idx = [j for j in range(n_pred) if j not in matched_pred]

    return MatchResult(
        matched_pairs=matched_pairs,
        unmatched_pred_idx=unmatched_pred_idx,
        unmatched_gt_idx=unmatched_gt_idx,
        iou_matrix=iou_matrix,
    )