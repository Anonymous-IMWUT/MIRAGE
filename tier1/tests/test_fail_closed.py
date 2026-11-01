#!/usr/bin/env python3
"""CONTRACT tests for the fail-closed box mask, on synthetic frames with a scripted detector.

    python -m pytest tier1/tests/test_fail_closed.py -q
    python tier1/tests/test_fail_closed.py            # same checks, no pytest needed

No models, no footage. `Scene` is a textured background that slides by a known step per frame, and
the detector is a dict of frame -> boxes, so every rule of the policy can be stated exactly.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from mirage.fail_closed import HOLD, POSE_CONF, FailClosedMask  # noqa: E402

W, H = 320, 240
BOX = [100.0, 60.0, 160.0, 180.0]


class Scene:
    def __init__(self, n, k, script, step=0):
        rng = np.random.RandomState(20261031)
        tex = rng.randint(0, 255, (H // 4, (W + 8 * n) // 4, 3), np.uint8)
        self.tex = cv2.resize(tex, (W + 8 * n, H), interpolation=cv2.INTER_CUBIC)
        self.n, self.k, self.script, self.step = n, k, script, step
        self.read_at = 0
        self.calls = 0

    def read(self):
        if self.read_at >= self.n:
            return None
        x = 4 * self.n - self.step * self.read_at
        frame = self.tex[:, x:x + W].copy()
        frame[0, 0, 0] = self.read_at          # lets a test check the capture order
        self.read_at += 1
        return frame

    def detect(self, frame):
        boxes = self.script.get(self.calls * self.k, [])
        self.calls += 1
        if not boxes:
            return np.empty((0, 4)), np.empty(0)
        return (np.array([b[:4] for b in boxes], float), np.array([b[4] for b in boxes], float))


def run(n, k, script, step=0):
    scene = Scene(n, k, script, step)
    fc = FailClosedMask(scene.detect, scene.read, W, H, k)
    out = []
    while True:
        item = fc.next()
        if item is None:
            return out, fc, scene
        out.append(item)


def covered(mask, box=BOX):
    x1, y1, x2, y2 = (int(v) for v in box)
    return mask is not None and bool(mask[(y1 + y2) // 2, (x1 + x2) // 2])


def test_every_frame_comes_back_in_capture_order():
    for k in (1, 2, 6):
        for n in (1, 5, 6, 7, 12, 13):
            out, fc, scene = run(n, k, {})
            assert [int(f[0, 0, 0]) for f, _, _ in out] == list(range(n)), (k, n)
            assert scene.calls == fc.detector_calls == (n + k - 1) // k, (k, n)
            assert fc.frames == n


def test_nobody_found_means_no_mask():
    out, _, _ = run(13, 6, {})
    assert all(m is None for _, m, _ in out)


def test_pose_gets_only_its_own_threshold_and_only_on_detector_frames():
    low = [10.0, 10.0, 50.0, 80.0, POSE_CONF - 0.01]
    high = BOX + [POSE_CONF]
    out, _, _ = run(12, 6, {0: [low, high], 6: [low]})
    for i, (_, _, boxes) in enumerate(out):
        if i % 6:
            assert boxes is None
    assert out[0][2].tolist() == [BOX]
    assert len(out[6][2]) == 0
    assert covered(out[0][1], low[:4])            # the mask still takes the low box


def test_two_way_covers_a_person_first_found_at_the_next_detection():
    out, _, _ = run(18, 6, {6: [BOX + [0.9]]})
    assert not covered(out[0][1])                 # a detector frame gets no back-fill
    for i in range(1, 7):
        assert covered(out[i][1]), i


def test_a_box_lost_at_one_detection_is_held_exactly_hold_cycles():
    k = 3
    out, _, _ = run(20, k, {0: [BOX + [0.9]]})
    last = (HOLD + 1) * k - 1
    for i, (_, mask, _) in enumerate(out):
        assert covered(mask) == (i <= last), i


def test_a_box_matched_by_the_next_detection_is_not_held():
    moved = [104.0, 60.0, 164.0, 180.0]
    out, _, _ = run(12, 3, {0: [BOX + [0.9]], 3: [moved + [0.9]], 6: [moved + [0.9]], 9: [moved + [0.9]]})
    x1, y1, x2, y2 = (int(v) for v in moved)
    assert int(out[9][1].sum()) == (x2 - x1 + 1) * (y2 - y1 + 1)


def test_boxes_follow_the_camera_both_ways():
    step = 4
    out, _, _ = run(12, 6, {0: [BOX + [0.9]]}, step=step)
    for i in range(1, 6):
        left = np.flatnonzero(out[i][1].any(0))[0]
        assert abs(left - (BOX[0] + step * i)) <= 2, (i, left)
    out, _, _ = run(12, 6, {6: [BOX + [0.9]]}, step=step)
    for i in range(1, 6):
        left = np.flatnonzero(out[i][1].any(0))[0]
        assert abs(left - (BOX[0] - step * (6 - i))) <= 2, (i, left)


def test_last_frames_after_the_last_detection_keep_the_forward_carry():
    out, _, _ = run(10, 6, {0: [], 6: [BOX + [0.9]]})
    assert [covered(m) for _, m, _ in out[6:]] == [True] * 4


def test_a_single_frame_interval_still_holds():
    out, _, _ = run(8, 1, {0: [BOX + [0.9]]})
    assert [covered(m) for _, m, _ in out] == [True] * (HOLD + 1) + [False] * (8 - HOLD - 1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print(f"{len(tests)} passed")
