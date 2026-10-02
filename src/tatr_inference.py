"""
Step 1 (TATR half) + Step 2: load TATR and run inference, converting its
raw row/column/spanning-cell predictions into the same unified schema.Table
used by the GT loaders.

Requires: pip install transformers torch pillow --break-system-packages
Requires network access to huggingface.co, which this sandbox does not have
-- run this file in your own environment, not here.
"""

from typing import List
from PIL import Image
from schema import Cell, Table

MODEL_NAME = "microsoft/table-structure-recognition-v1.1-all"


class TATRPredictor:
    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        # Imports kept inside __init__ so the rest of the pipeline (schema,
        # matching, error_extraction) can be imported/tested without torch
        # or transformers installed.
        import torch
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.processor.size = {
    "shortest_edge": 800,
    "longest_edge": 1000
}
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name).to(device)
        self.model.eval()
        self.torch = torch
        self.id2label = self.model.config.id2label

    def predict_raw(self, image: Image.Image, score_threshold: float = 0.6):
        """Returns TATR's raw per-object detections: label, box, score."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": self.id2label[int(label_id)],
                "score": float(score),
                "box": [float(x) for x in box.tolist()],  # xmin, ymin, xmax, ymax
            })
        return detections

    def predict_table(self, image: Image.Image, image_id: str,
                       score_threshold: float = 0.6) -> Table:
        """
        Converts raw row/column/spanning-cell detections into a grid of
        Cells, the same way loaders.load_pubtables1m reconstructs cells
        from GT row/column boxes -- so predicted and GT tables are directly
        comparable by the matching/error_extraction code.
        """
        detections = self.predict_raw(image, score_threshold)

        rows = [d["box"] for d in detections if d["label"] == "table row"]
        cols = [d["box"] for d in detections if d["label"] == "table column"]
        spanning = [d["box"] for d in detections if d["label"] == "table spanning cell"]
        headers = [d["box"] for d in detections if d["label"] == "table column header"]

        rows.sort(key=lambda b: b[1])
        cols.sort(key=lambda b: b[0])

        def row_overlaps(box, row_box):
            return not (box[3] <= row_box[1] or box[1] >= row_box[3])

        def col_overlaps(box, col_box):
            return not (box[2] <= col_box[0] or box[0] >= col_box[2])

        header_row_idxs = {ri for ri, r in enumerate(rows)
                            for h in headers if row_overlaps(h, r)}

        cells: List[Cell] = []
        for ri, r in enumerate(rows):
            for ci, c in enumerate(cols):
                bbox = [c[0], r[1], c[2], r[3]]
                cells.append(Cell(bbox=bbox, start_row=ri, end_row=ri,
                                   start_col=ci, end_col=ci,
                                   is_header=(ri in header_row_idxs)))

        for sc in spanning:
            covered_rows = [ri for ri, r in enumerate(rows) if row_overlaps(sc, r)]
            covered_cols = [ci for ci, c in enumerate(cols) if col_overlaps(sc, c)]
            if not covered_rows or not covered_cols:
                continue
            cells = [c for c in cells if not (
                c.start_row in covered_rows and c.start_col in covered_cols)]
            cells.append(Cell(bbox=sc, start_row=min(covered_rows), end_row=max(covered_rows),
                               start_col=min(covered_cols), end_col=max(covered_cols),
                               is_header=any(r in header_row_idxs for r in covered_rows)))

        return Table(image_id=image_id, dataset_source="TATR_prediction",
                     cells=cells, image_width=image.width, image_height=image.height)