"""
Validates matching.py + error_extraction.py against synthetic tables where
we know exactly which errors were injected. No TATR model or real dataset
needed -- this proves the core logic (the part that doesn't depend on
dataset-specific annotation formats) is correct.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema import Cell, Table
from schema import Cell, Table
from error_extraction import extract_errors

W, H = 40, 20  # cell width/height for the synthetic grid


def grid_cell(row, col, start_row=None, end_row=None, start_col=None, end_col=None,
              is_header=False):
    start_row = row if start_row is None else start_row
    end_row = row if end_row is None else end_row
    start_col = col if start_col is None else start_col
    end_col = col if end_col is None else end_col
    x1, y1 = start_col * W, start_row * H
    x2, y2 = (end_col + 1) * W, (end_row + 1) * H
    return Cell(bbox=[x1, y1, x2, y2], start_row=start_row, end_row=end_row,
                start_col=start_col, end_col=end_col, is_header=is_header)


def make_clean_3x3():
    """A perfect 3x3 grid, row 0 is the header. No errors."""
    cells = [grid_cell(r, c, is_header=(r == 0)) for r in range(3) for c in range(3)]
    return Table(image_id="t", dataset_source="synthetic", cells=cells)


def test_no_errors_on_identical_tables():
    gt = make_clean_3x3()
    pred = make_clean_3x3()
    result = extract_errors(gt, pred)
    assert result["has_errors"] is False, result
    assert result["error_count"] == 0
    print("PASS: identical tables -> 0 errors")


def test_missing_cell():
    gt = make_clean_3x3()
    pred_cells = [c for c in gt.cells if not (c.start_row == 1 and c.start_col == 1)]
    pred = Table(image_id="t", dataset_source="pred", cells=pred_cells)
    result = extract_errors(gt, pred)
    assert len(result["errors_found"]["missing_cells"]) == 1
    print("PASS: missing cell detected")


def test_extra_cell_hallucination():
    gt = make_clean_3x3()
    hallucinated = Cell(bbox=[500, 500, 540, 520], start_row=9, end_row=9, start_col=9, end_col=9)
    pred = Table(image_id="t", dataset_source="pred", cells=gt.cells + [hallucinated])
    result = extract_errors(gt, pred)
    assert len(result["errors_found"]["extra_cells"]) == 1
    print("PASS: hallucinated extra cell detected")


def test_incorrect_merge():
    """TATR predicts one wide cell covering GT's (1,0) and (1,1) -- undersegmentation."""
    gt = make_clean_3x3()
    pred_cells = [c for c in gt.cells if not (c.start_row == 1 and c.start_col in (0, 1))]
    merged = Cell(bbox=[0 * W, 1 * H, 2 * W, 2 * H], start_row=1, end_row=1, start_col=0, end_col=1)
    pred = Table(image_id="t", dataset_source="pred", cells=pred_cells + [merged])
    result = extract_errors(gt, pred)
    assert len(result["errors_found"]["incorrect_merges"]) == 1, result["errors_found"]
    merge = result["errors_found"]["incorrect_merges"][0]
    assert len(merge["merged_gt_bboxes"]) == 2
    print("PASS: incorrect merge detected")


def test_incorrect_split():
    """TATR predicts GT's single (0,0) header cell as two side-by-side half-width cells."""
    gt = make_clean_3x3()
    pred_cells = [c for c in gt.cells if not (c.start_row == 0 and c.start_col == 0)]
    half_a = Cell(bbox=[0, 0, W / 2, H], start_row=0, end_row=0, start_col=0, end_col=0, is_header=True)
    half_b = Cell(bbox=[W / 2, 0, W, H], start_row=0, end_row=0, start_col=0, end_col=0, is_header=True)
    pred = Table(image_id="t", dataset_source="pred", cells=pred_cells + [half_a, half_b])
    result = extract_errors(gt, pred)
    assert len(result["errors_found"]["incorrect_splits"]) == 1, result["errors_found"]
    print("PASS: incorrect split detected")


def test_row_span_error_and_header_flag():
    """
    GT has a normal (0,0) header cell. TATR predicts the same bbox location
    but reports it as spanning rows 0-1 (wrong span). Since row 0 is a GT
    header row, this should also be flagged as a header_alignment_error.
    """
    gt = make_clean_3x3()
    pred_cells = [c for c in gt.cells if not (c.start_row == 0 and c.start_col == 0)]
    wrong_span = Cell(bbox=[0, 0, W, 2 * H], start_row=0, end_row=1, start_col=0, end_col=0, is_header=True)
    pred = Table(image_id="t", dataset_source="pred", cells=pred_cells + [wrong_span])
    result = extract_errors(gt, pred)
    span_errors = result["errors_found"]["row_col_relationship_errors"]
    assert len(span_errors) == 1, span_errors
    assert span_errors[0]["row_span_error"] is True
    header_errors = result["errors_found"]["header_alignment_errors"]
    # A wrong row-span on a header cell legitimately overlaps 2 GT cells, so
    # it is correctly caught by BOTH the span-error rule and the merge rule --
    # two different lenses on the same underlying mistake. Both must be
    # present and both must carry the header flag.
    error_types = {e["error_type"] for e in header_errors}
    assert error_types == {"span_error", "incorrect_merge"}, header_errors
    assert all(e["header_alignment_error"] is True for e in header_errors)
    print("PASS: row span error detected and correctly flagged as header error "
          "(both as a span error and as the merge it also constitutes)")


def test_col_span_error_not_flagged_as_header_when_body_row():
    """Same span-error mechanics but in row 2 (a body row) -- must NOT be header-flagged."""
    gt = make_clean_3x3()
    pred_cells = [c for c in gt.cells if not (c.start_row == 2 and c.start_col == 0)]
    wrong_span = Cell(bbox=[0, 2 * H, 2 * W, 3 * H], start_row=2, end_row=2, start_col=0, end_col=1)
    pred = Table(image_id="t", dataset_source="pred", cells=pred_cells + [wrong_span])
    result = extract_errors(gt, pred)
    header_errors = result["errors_found"]["header_alignment_errors"]
    assert len(header_errors) == 0, header_errors
    print("PASS: body-row span/merge correctly NOT flagged as header error")


if __name__ == "__main__":
    test_no_errors_on_identical_tables()
    test_missing_cell()
    test_extra_cell_hallucination()
    test_incorrect_merge()
    test_incorrect_split()
    test_row_span_error_and_header_flag()
    test_col_span_error_not_flagged_as_header_when_body_row()
    print("\nAll tests passed.")