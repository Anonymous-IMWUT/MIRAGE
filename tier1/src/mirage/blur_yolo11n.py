import cv2
import numpy as np
import os

HULL_COLOR = (127, 127, 127)


class YOLO11nBoxBlur:
    """
    Plain (detection-only) YOLO11n, NCNN backend -- NOT yolo11n-seg. No
    segmentation mask: the full detector box is grey-filled outright.

    Rationale (2026-08-03/04 sessions): dropping the segmentation mask
    removes a real re-id side-channel -- a mask-shaped grey cutout still
    leaks the person's outline/silhouette (a known gait/body-shape re-id
    signal, see project memory on the topic), while a rectangular box
    grey-fill destroys that outline entirely. Detector accuracy comparison
    (2026-08-04, eval_detector_compare.py, full 16,507-frame human-verified
    GT sweep) showed YOLO11n-NCNN beats YOLOX-Nano on recall in every one
    of 11 dataset categories -- most notably MovementHead (+7.7pt) and the
    hardest multi-person cases NumFaces4/5 (+3.9/+4.6pt) -- at a real but
    bounded ~1.65x per-frame speed cost (141ms vs 85ms/frame on laptop CPU,
    unthreaded). This class exists to let that detector swap be benchmarked
    through the REAL threaded pipeline (skip-frame cadence, mask-warp
    propagation, thread pools), not just the standalone comparison scripts.

    Deliberately SAME PUBLIC INTERFACE as YOLOSegBlur (get_mask_and_boxes,
    get_mask, apply_mask) so it's a drop-in replacement wherever pipeline.py
    references `yolo_seg` -- no changes needed to skip-frame mask-warp
    propagation, apply_mask() call sites, or export-mode mask writers. The
    "mask" returned here is just a rasterized rectangle (bool H×W, True
    inside each detected box) instead of a real per-pixel segmentation --
    every downstream consumer only ever treats it as an opaque bool array,
    so this satisfies the contract exactly.
    """

    def __init__(self, model_name: str, infer_size: int = 320, conf: float = 0.4):
        from ultralytics import YOLO
        self._model      = YOLO(model_name)
        self._infer_size = infer_size
        self._conf       = conf

    def get_mask(self, frame: np.ndarray, infer_size: int = None):
        """Run YOLO11n on frame, return a box-rasterized bool mask (H×W).
        infer_size arg accepted for API parity with YOLOSegBlur; ignored.
        Returns None if no persons detected."""
        mask, _ = self.get_mask_and_boxes(frame)
        return mask

    def get_mask_and_boxes(self, frame: np.ndarray):
        """
        Same (mask_or_None, boxes) contract as YOLOSegBlur.get_mask_and_boxes()
        -- boxes is an (N,4) float array [x1,y1,x2,y2] in full-frame pixel
        space (or empty if none detected). mask is the SAME boxes rasterized
        into a bool H×W array (union of all detected boxes) rather than a
        real segmentation mask -- see class docstring for why.
        """
        h, w = frame.shape[:2]
        results = self._model(
            frame,
            imgsz=self._infer_size,
            conf=self._conf,
            classes=[0],
            verbose=False,
        )
        combined = np.zeros((h, w), dtype=bool)
        boxes = np.empty((0, 4), dtype=float)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
        for b in boxes:
            x1, y1, x2, y2 = (int(v) for v in b)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                combined[y1:y2, x1:x2] = True
        mask = combined if combined.any() else None
        return mask, boxes

    def get_boxes_and_conf(self, frame: np.ndarray):
        """Every box at or above this detector's threshold, with its score, in
        full-frame pixels. The fail-closed mask rasterizes boxes itself and needs
        the scores to give pose its own higher threshold."""
        r = self._model(
            frame,
            imgsz=self._infer_size,
            conf=self._conf,
            classes=[0],
            verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 4), dtype=float), np.empty(0, dtype=float)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[mask] = HULL_COLOR
        return out

    def blur_frame(self, frame: np.ndarray) -> np.ndarray:
        mask = self.get_mask(frame)
        return self.apply_mask(frame, mask) if mask is not None else frame
