"""
Unified schema used across ground-truth loaders, TATR inference output,
and the error-extraction pipeline.

Every dataset (PubTables-1M, FinTabNet, SciTSR, or TATR's own predictions)
gets converted into this same shape so the rest of the pipeline never has
to know which dataset a table came from.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Cell:
    """One logical table cell."""
    bbox: Optional[List[float]]  # [xmin, ymin, xmax, ymax] in image pixel coords.
                                  # None for cells where no bbox could be recovered
                                  # (e.g. SciTSR cells with no text chunk at all --
                                  # such cells are excluded from IoU-based matching
                                  # upstream; this field exists so callers can still
                                  # see they existed and were skipped, rather than
                                  # silently vanishing.)
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    text_content: str = ""
    is_header: bool = False    # True if this cell belongs to a header row/col

    def row_span(self) -> int:
        return self.end_row - self.start_row + 1

    def col_span(self) -> int:
        return self.end_col - self.start_col + 1


@dataclass
class Table:
    """One table: a list of cells plus light metadata."""
    image_id: str
    dataset_source: str
    cells: List[Cell] = field(default_factory=list)          # only cells WITH a bbox
    image_width: Optional[float] = None
    image_height: Optional[float] = None
    skipped_no_bbox_count: int = 0   # cells that exist in GT but had no bbox available
                                      # (e.g. genuinely empty SciTSR cells) -- not
                                      # included in `cells`, not seen by matching.py

    def header_row_indices(self) -> set:
        return {c.start_row for c in self.cells if c.is_header} | \
               {c.end_row for c in self.cells if c.is_header}


def cell_to_dict(c: Cell) -> Dict[str, Any]:
    return asdict(c)


def table_to_dict(t: Table) -> Dict[str, Any]:
    d = asdict(t)
    return d


def dict_to_cell(d: Dict[str, Any]) -> Cell:
    return Cell(
        bbox=list(d["bbox"]),
        start_row=int(d["start_row"]),
        end_row=int(d["end_row"]),
        start_col=int(d["start_col"]),
        end_col=int(d["end_col"]),
        text_content=d.get("text_content", ""),
        is_header=bool(d.get("is_header", False)),
    )


def dict_to_table(d: Dict[str, Any]) -> Table:
    return Table(
        image_id=d["image_id"],
        dataset_source=d.get("dataset_source", "unknown"),
        cells=[dict_to_cell(c) for c in d.get("cells", [])],
        image_width=d.get("image_width"),
        image_height=d.get("image_height"),
    )