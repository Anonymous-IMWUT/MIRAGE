"""Fail-closed box mask for yolo11n_boxfill: a low mask threshold, a two-way fill and a hold.

A box mask carried forward from a detector frame cannot cover a person the detector scored low,
lost for one cycle, or first found at the NEXT detector frame. With the detector on every sixth
frame, those gaps are where faces stayed visible on head-motion footage. Three rules close them:

  threshold  The mask takes every box at or above the detector's own threshold (0.05 when this is
             on). Pose still gets only the boxes at or above POSE_CONF, so the skeleton does not
             pick up false boxes.
  two-way    The next detection is warped back over the gap and unioned with the forward carry.
             A frame is released only once the next detector pass has run, so the mask is k frames
             behind the camera.
  hold       A box with no match at the next detection stays for HOLD more detector cycles.

Boxes follow one global affine fitted to a background point grid on EVERY frame, not keypoints, so
the mask keeps working for people who were never tracked and across frames with no pose at all.
"""
import collections
import time

import cv2
import numpy as np

from .pose import LK_PARAMS

POSE_CONF  = 0.40       # pose keeps the threshold it was measured at
HOLD       = 2          # detector cycles an unmatched box survives
MATCH_IOU  = 0.3        # a held box is dropped once a new box overlaps it this much
WORK_SIZE  = 640        # long side of the LK working frame; translation is scaled back up
GRID       = 14         # points per side of the background grid
MIN_POINTS = 8          # tracked points below which no affine is fitted


def box_corners(boxes):
    """(N,4) xyxy boxes -> (N,4,2) corners, so a warped box can rotate."""
    if len(boxes) == 0:
        return np.empty((0, 4, 2), np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([np.stack([x1, y1], 1), np.stack([x2, y1], 1),
                     np.stack([x2, y2], 1), np.stack([x1, y2], 1)], 1).astype(np.float32)


def warp_corners(corners, M):
    if len(corners) == 0:
        return corners
    pts = np.concatenate([corners, np.ones(corners.shape[:2] + (1,), np.float32)], 2)
    return pts @ M.T


def invert_affine(M):
    return cv2.invertAffineTransform(np.asarray(M, np.float64)).astype(np.float32)


def _aabb(corners):
    return np.concatenate([corners.min(1), corners.max(1)], 1)


def _overlaps(a, b):
    """For each box in a, does any box in b overlap it at IoU >= MATCH_IOU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), bool)
    lo = np.maximum(a[:, None, :2], b[None, :, :2])
    hi = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(hi - lo, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)[:, None]
    area_b = (b[:, 2:] - b[:, :2]).prod(1)[None]
    return (inter / np.maximum(area_a + area_b - inter, 1e-6)).max(1) >= MATCH_IOU


class EgoMotion:
    """Frame-to-frame global affine, in full-resolution pixels."""

    def __init__(self, width, height):
        self._scale = WORK_SIZE / max(width, height)
        self._size = (int(round(width * self._scale)), int(round(height * self._scale)))
        xs = np.linspace(0.06, 0.94, GRID) * self._size[0]
        ys = np.linspace(0.06, 0.94, GRID) * self._size[1]
        self._pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 1, 2).astype(np.float32)
        self._prev = None

    def step(self, frame):
        """Affine from the previous frame to this one; None on the first frame."""
        small = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self._prev = self._prev, gray
        if prev is None:
            return None
        new, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, self._pts, None, **LK_PARAMS)
        good = status.ravel() == 1
        M = None
        if good.sum() >= MIN_POINTS:
            M, _ = cv2.estimateAffinePartial2D(self._pts[good], new[good], method=cv2.RANSAC,
                                               ransacReprojThreshold=2.0)
        if M is None:
            return np.array([[1, 0, 0], [0, 1, 0]], np.float32)   # no estimate: boxes stay put
        M = M.copy()
        M[:, 2] /= self._scale
        return M.astype(np.float32)


class FailClosedMask:
    """Sits between the read queue and the processing loop, one detector gap ahead of it.

    `read()` returns the next frame or None at end of stream. `detect(frame)` returns (boxes, conf)
    in full-frame pixels, already cut at the mask threshold. next() hands back (frame, mask,
    pose_boxes) in capture order: mask is a bool H x W array or None, pose_boxes is the POSE_CONF
    subset on detector frames and None on every other frame.
    """

    def __init__(self, detect, read, width, height, k):
        self.k = int(k)
        self._detect = detect
        self._read = read
        self._shape = (height, width)
        self._ego = EgoMotion(width, height)
        self._ready = collections.deque()
        self._anchor = None                                 # detector frame the next gap starts at
        self._live = np.empty((0, 4, 2), np.float32)        # boxes carried into the next detection
        self._age = np.empty(0, int)
        self._eos = False
        self.frames = 0
        self.detector_calls = 0
        self.t_detect = 0.0
        self.t_ego = 0.0

    def next(self):
        while not self._ready and not self._eos:
            self._advance()
        return self._ready.popleft() if self._ready else None

    def stats(self):
        return {"detector_every": self.k, "pose_conf": POSE_CONF, "hold": HOLD, "two_way": True,
                "frames": self.frames, "detector_calls": self.detector_calls}

    def _pull(self):
        """Next frame and the affine that reaches it, or (None, None) at end of stream."""
        frame = self._read()
        if frame is None:
            self._eos = True
            return None, None
        t0 = time.time()
        M = self._ego.step(frame)
        self.t_ego += time.time() - t0
        return frame, M

    def _run_detector(self, frame):
        t0 = time.time()
        boxes, conf = self._detect(frame)
        self.t_detect += time.time() - t0
        self.detector_calls += 1
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        return frame, box_corners(boxes), boxes[np.asarray(conf) >= POSE_CONF]

    def _carry(self, fresh):
        """This detection plus whatever is still held; updates the live set and its ages."""
        live, age = self._live, self._age
        if len(live):
            keep = ~_overlaps(_aabb(live), _aabb(fresh)) & (age < HOLD)
            live, age = live[keep], age[keep] + 1
        self._live = np.concatenate([fresh, live])
        self._age = np.concatenate([np.zeros(len(fresh), int), age])
        return self._live

    def _raster(self, parts):
        ink = np.zeros(self._shape, np.uint8)
        for corners in parts:
            for c in corners:
                cv2.fillConvexPoly(ink, np.round(c).astype(np.int32), 1)
        mask = ink.astype(bool)
        return mask if mask.any() else None

    def _advance(self):
        """Read one gap ahead, then settle and queue every frame before the next detection."""
        if self._anchor is None:
            frame, _ = self._pull()
            if frame is None:
                return
            self._anchor = self._run_detector(frame)
        anchor, fresh, pose_boxes = self._anchor
        gap, aff = [anchor], []
        for _ in range(self.k):
            frame, M = self._pull()
            if frame is None:
                break
            gap.append(frame)
            aff.append(M)
        following = self._run_detector(gap.pop()) if len(gap) == self.k + 1 else None
        self._anchor = following

        forward = [self._carry(fresh)]
        for M in aff[:len(gap) - 1]:
            forward.append(warp_corners(forward[-1], M))
        parts = [[c] for c in forward]
        if following is not None:
            back = following[1]
            for i in range(len(gap) - 1, 0, -1):
                back = warp_corners(back, invert_affine(aff[i]))
                parts[i].append(back)
            self._live = warp_corners(forward[-1], aff[-1])

        for i, frame in enumerate(gap):
            self._ready.append((frame, self._raster(parts[i]), pose_boxes if i == 0 else None))
        self.frames += len(gap)
