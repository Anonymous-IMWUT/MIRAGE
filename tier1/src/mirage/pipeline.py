import cv2
import sys
import time
import os
import csv
import json
import uuid
import queue
import threading
import urllib.request
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from rtmlib import RTMPose, draw_skeleton

from .pose import (
    euclidean, get_face_size_tier, get_movement_tier,
    derive_face_crop, derive_body_crop, compute_frame_confidence,
    project_landmarks, draw_face_mesh_pts,
    COCO_NOSE, COCO_LEFT_EYE, COCO_RIGHT_EYE, LK_PARAMS,
)
from .blur import blur_all_persons
from .blur_seg import SelfieSegBlur, bbox_region_mask
from .blur_mobilesam import MobileSAMBlur, bboxes_from_keypoints
from .blur_yoloseg import YOLOSegBlur
from .blur_yolo11n import YOLO11nBoxBlur
from .fail_closed import FailClosedMask, HOLD as FC_HOLD, POSE_CONF as FC_POSE_CONF
from .face_canonical import FaceCanonicalizer, CANONICAL_SIZE, yaw_from_transform, face_quality_from_yaw
from .face_canonical_v2 import FaceCanonicalizerV2, P_SMILE
from .tracking import PersonState
from .export_tracking import PersonIdentityTracker
from .encryption import fetch_ttp_public_key
from .embedding import EmbeddingExtractor, EDGEFACE_ONNX_PATH
from .gender import GenderClassifier, GENDER_ONNX_PATH

BASE_RESOLUTION       = 1280.0
BASE_FAR_THRESHOLD    = 30
BASE_MEDIUM_THRESHOLD = 80
BASE_SLOW_THRESHOLD   = 5
BASE_FAST_THRESHOLD   = 15

SKIP_N_DEFAULTS = {
    "slow":   7,
    "medium": 4,
    "fast":   1,
}

FACE_MESH_MIN_CONF = 0.3
INFER_SIZE         = 320
TIMING_INTERVAL    = 30
DEBUG_DRAW         = True

# Consecutive full frames a stale selfie-seg gate region may be reused for
# when det+pose both fail to produce any usable bbox on a given frame (e.g.
# transient motion-blur pose dropout). Small enough that a person who truly
# leaves the frame stops being blurred within a fraction of a second.
GATE_REGION_TTL = 5

# Minimum ratio (this box's area / largest box's area in the same frame) for
# a detection to be treated as a trackable person, when more than one
# box is detected. Filters out distant background bystanders relative to
# whoever is dominant/closest to camera in that frame, without relying on a
# fragile absolute pixel-size cutoff (subject box size varies a lot with
# distance from camera across a clip).
MIN_BOX_AREA_RATIO = 0.20

# --- MIRAGE re-ID defence support (see the defence kwargs of process_video) ---
# Longest side of the downscaled emitted-mask buffer. The buffer exists only to
# answer "did the exported stick figure land inside the grey region", which is a
# coverage question, not a boundary question -- 256 px is enough for it and keeps
# a whole clip in memory. Not a tuned value: nothing is fitted to it, and the
# metric reports the resolution it was measured at.
MASK_BUF_MAX_DIM = 256

# COCO-17 skeleton links, used ONLY to rasterize "stick ink" for the containment
# metric below. This is the standard COCO topology (the same edge set rtmlib's
# draw_skeleton uses, 0-indexed); it is not a rendering choice that reaches any
# output frame.
COCO17_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
)


def _stick_ink_containment(kp_arr, masks, sx, sy, thresh):
    """How much of one slot's EXPORTED skeleton falls inside the EMITTED mask.

    This is the check that would have caught RUN3's c2 failure on the MIRAGE
    host (ledger B.61a/b): the generator obeyed the mask perfectly, but the
    skeleton it was given was not where the repaintable hole was -- 49 of 140
    frames had ZERO stick ink inside the mask. A per-frame mean alone hides
    that, which is why `min` and the zero-count are reported beside it.

    `kp_arr` is the slot's FINAL exported array (T, 17, 3) -- post-gait,
    post-binarize -- so this measures what actually reaches disk, not an
    intermediate. `masks` are the downscaled emitted masks, `sx`/`sy` the
    native->downscaled scale factors, `thresh` the confidence at/above which a
    joint is drawable.

    Ink that falls OUTSIDE the frame is counted as NOT contained, which is the
    conservative direction. cv2.line would silently clip such a segment away
    (inflating containment), so the scratch canvas is padded and coordinates are
    clamped into the pad rather than dropped.

    Returns None when there was nothing to measure.
    """
    n = min(len(kp_arr), len(masks))
    if n == 0:
        return None
    mh, mw   = masks[0].shape
    pad      = max(1, max(mh, mw) // 4)
    ink      = np.zeros((mh + 2 * pad, mw + 2 * pad), dtype=np.uint8)
    field    = np.zeros(ink.shape, dtype=bool)
    vals, zero, absent, no_ink = [], 0, 0, 0
    for t in range(n):
        row = kp_arr[t]
        if not row.any():
            absent += 1            # slot empty this frame (np.zeros marker)
            continue
        ink[:] = 0
        drew = False
        for a, b in COCO17_SKELETON:
            if float(row[a, 2]) < thresh or float(row[b, 2]) < thresh:
                continue
            pa = (int(np.clip(round(float(row[a, 0]) * sx) + pad, 0, ink.shape[1] - 1)),
                  int(np.clip(round(float(row[a, 1]) * sy) + pad, 0, ink.shape[0] - 1)))
            pb = (int(np.clip(round(float(row[b, 0]) * sx) + pad, 0, ink.shape[1] - 1)),
                  int(np.clip(round(float(row[b, 1]) * sy) + pad, 0, ink.shape[0] - 1)))
            cv2.line(ink, pa, pb, 1, 1)
            drew = True
        total = int(np.count_nonzero(ink)) if drew else 0
        if total == 0:
            no_ink += 1            # nothing confident enough to draw
            continue
        field[:] = False
        field[pad:pad + mh, pad:pad + mw] = masks[t]
        inside = int(np.count_nonzero((ink != 0) & field))
        vals.append(inside / float(total))
        if inside == 0:
            zero += 1
    return {
        "frames_measured":        len(vals),
        "mean":                   (round(float(np.mean(vals)), 6) if vals else None),
        "min":                    (round(float(np.min(vals)), 6) if vals else None),
        "zero_containment_frames": zero,
        "frames_slot_absent":     absent,
        "frames_no_drawable_ink": no_ink,
        "ink_conf_thresh":        float(thresh),
        "measured_at":            f"{mw}x{mh} px (emitted mask, downscaled)",
        "note": ("fraction of the drawn COCO-17 skeleton's pixels lying inside the "
                 "EMITTED (post-mitigation) mask; ink outside the frame counts as "
                 "outside. Frames where the slot was absent, or where no link had "
                 "both endpoints at/above ink_conf_thresh, are excluded from mean/min "
                 "and counted separately."),
    }


def process_video(
    input_path,
    output_path       = "/tmp/output_rtm.mp4",
    blur_bodies       = True,
    enc_output_dir    = "data/output/encrypted",
    headless          = False,
    save_video        = True,
    benchmark         = False,
    skip_n            = 5,
    movement_adaptive = False,
    csv_out           = None,
    ttp_server        = None,   # Tier 3 TTP base URL, e.g. "https://localhost:8843" -- required unless benchmark=True
    ttp_verify_tls    = True,   # False for --ttp-http / TOFU-unverified local testing
    anonymizer        = "convexhull",   # "convexhull" | "selfie_seg0" | "selfie_seg1" | "mobilesam" | "yoloseg" | "yoloseg11" | "yoloseg11int8" | "yoloseg11ncnn"
    export_dir         = None,   # dense per-person export mode (opt-in, additive -- see export_tracking.py)
    dense_export        = False,
    export_people        = 3,
    export_diagnostics  = False,
    seg_infer_size      = 320,   # yoloseg* network input size (px); lower = faster, coarser masks
    seg_skip_n          = 1,     # yoloseg* segmentation cadence, independent of skip_n (1 = every frame)
    no_draw             = False, # suppress skeleton/facemesh debug overlay (and HUD text) entirely
    no_facemesh_draw    = False, # suppress only the 468-pt facemesh overlay; skeleton still drawn
    no_hud              = False, # suppress only the FPS/Frame/People/Movement/Skip-N corner text
    # ------------------------------------------------------------------------
    # MIRAGE re-ID defences (vendored byte-identical under vendor/mirage_edge/;
    # adapters in gait_anon.py / mask_anon.py / provenance.py).
    #
    # BOTH DEFAULT OFF, and every line that implements them lives inside an
    # `if <enabled>:` guard -- including the imports, which are lazy. A run
    # that leaves these at their defaults executes exactly the code this file
    # executed before they existed, and never touches the vendored package.
    #
    # 🔴 NO PRIVACY NUMBER MEASURED ON THE MIRAGE HOST DESCRIBES THIS HOST.
    #    The preset/mode names are recorded so a bundle is self-identifying;
    #    that is a provenance claim, not a re-ID claim.
    # ------------------------------------------------------------------------
    gait_anon             = True,    # whole-clip gait transform on the EXPORTED keypoints --
                                     # default ON: the paper's Section 4 states
                                     # this transform runs before egress, unconditionally; pass
                                     # gait_anon=False (--no-gait-anon on the CLI) to reproduce
                                     # the paper's "Quantized Pose Baseline" ablation row.
    gait_preset           = "e2",    # vendored preset name; "" = bare LEVELS, no preset kwargs
    gait_level            = "L4",    # vendored LEVELS ladder entry
    gait_pin_run_seed     = False,   # TEST-ONLY pin; default is the vendored per-SEQUENCE draw
                                     # (owner decision 2026-08-14) -- see the seed block below
    gait_on_degenerate    = "raise", # "raise" | "skip" for degenerate bone lengths (policy)
    mask_shape_mode       = "none",  # "none" = off; "bbox" is the measured silhouette arm
    mask_temporal_win     = 2,       # PINNED to the arm A.6o measured bbox at (owner decision
                                     # 2026-08-14). None or 0 => derive from the emitted-mask fps.
    score_binarize        = None,    # None => follow gait_anon; True/False force it
    score_binarize_thresh = 0.5,     # owner decision: 0.5 == MIRAGE POSE_THRESH
    fail_closed           = False,   # fail-closed box mask, see fail_closed.py. yolo11n_boxfill only;
                                     # seg_skip_n is its detector interval.
    fail_closed_conf      = 0.05,    # mask threshold when fail_closed is on; pose keeps 0.40
):
    if benchmark:
        save_video = False
        headless   = True

    # Dense export needs literal every-frame accuracy (no skip-frame
    # optical-flow propagation) and a consistent segmentation backend to
    # populate mask.mp4/raw_seg_mask.mp4/gate_region.mp4. Both are forced
    # only when export_dir is actually set, so default (export_dir=None)
    # calls are entirely unaffected.
    export_enabled = export_dir is not None
    if export_enabled:
        dense_export = True
        os.makedirs(export_dir, exist_ok=True)
        # yolo11n_boxfill added to this whitelist 2026-08-14: it wasn't
        # included when first wired in earlier today, so every export-mode
        # run made with --anonymizer yolo11n_boxfill up to this point was
        # SILENTLY forced to selfie_seg1 instead -- confirmed via the log
        # line this block itself prints ("NOTE: export mode forces
        # anonymizer=..."), which is exactly what surfaced the bug.
        _export_ok_anonymizers = ("selfie_seg", "yoloseg", "yolo11n_boxfill")
        if not anonymizer.startswith(_export_ok_anonymizers):
            print(f"  NOTE: export mode forces anonymizer='selfie_seg1' (was '{anonymizer}')")
            anonymizer = "selfie_seg1"

    # --- MIRAGE silhouette defence: resolve the mode, refuse what cannot work -
    # Checked HERE -- after export mode has had its say on `anonymizer`, and
    # before any model is loaded -- so an unsupported combination fails in a
    # second instead of after a 30 s model load.
    #
    # convexhull and mobilesam are refused, not silently ignored: neither keeps
    # a persistent bool mask for this function to mitigate. blur_all_persons()
    # composites the grey fill straight onto the frame from the hull polygon,
    # and MobileSAMBlur.blur_frame() does its own compositing internally -- in
    # both cases `last_seg_mask` stays None for the whole clip, so a silhouette
    # defence wired at the mask seam would be a NO-OP that still reported
    # itself as enabled in the manifest. That is the exact "declared one config
    # while carrying another" failure the provenance module exists to prevent.
    mask_shape_mode = str(mask_shape_mode or "none").strip()
    mask_enabled    = mask_shape_mode.lower() not in ("", "none", "off")
    if mask_enabled and anonymizer in ("convexhull", "mobilesam"):
        raise ValueError(
            f"mask_shape_mode={mask_shape_mode!r} cannot be used with "
            f"anonymizer={anonymizer!r}. That backend composites the grey fill "
            f"directly onto the frame and never produces a persistent boolean "
            f"mask, so there is nothing at the mask seam to mitigate and the "
            f"defence would silently do nothing.\n"
            f"  Supported backends: selfie_seg0, selfie_seg1, yoloseg, "
            f"yoloseg11, yoloseg11int8, yoloseg11ncnn, yolo11n_boxfill.\n"
            f"  (The measured MIRAGE arm is a segmentation backend + the "
            f"'bbox' shape mode -- owner decision.)"
        )
    if mask_enabled:
        # 🔴 And refuse an unrecognised SHAPE MODE, for the same reason and here
        # for the same "fail in a second, not after a 30 s model load" reason.
        # The vendored `_shape_polys` has no else-branch: an unknown name falls
        # through to plain contour simplification, so a typo emits a nearly
        # unmitigated silhouette while the manifest declares the defence on
        # (measured: emitted/input area 1.082x for 'bbxo' vs 2.326x for 'bbox'
        # on the same input). The valid set is read off the vendored source.
        # Imported INSIDE the guard so the defences-off path still never touches
        # the vendored package.
        from .mask_anon import check_shape_mode as _check_shape_mode
        mask_shape_mode = _check_shape_mode(mask_shape_mode)

    # Same reasoning: the policy works on detector boxes, so on any other backend, or
    # with --no-blur, it would be declared on while doing nothing.
    if fail_closed and (anonymizer != "yolo11n_boxfill" or not blur_bodies):
        raise ValueError(
            f"fail_closed needs anonymizer='yolo11n_boxfill' with blurring on "
            f"(got anonymizer={anonymizer!r}, blur_bodies={blur_bodies})."
        )

    SKIP_N = SKIP_N_DEFAULTS.copy()
    if not movement_adaptive:
        SKIP_N["slow"]   = skip_n
        SKIP_N["medium"] = skip_n
        SKIP_N["fast"]   = skip_n
    if dense_export:
        SKIP_N = {"slow": 1, "medium": 1, "fast": 1}

    draw_enabled = DEBUG_DRAW and not benchmark and not no_draw

    print("=" * 60)
    print("  RTMPose Pipeline (Optical Flow + EdgeFace + Encryption)")
    print(f"  Infer size        : {INFER_SIZE}x{INFER_SIZE}")
    print(f"  Blur bodies       : {blur_bodies}")
    print(f"  Benchmark mode    : {benchmark}")
    print(f"  Movement adaptive : {movement_adaptive}")
    print(f"  Skip-N            : {SKIP_N}")
    if fail_closed:
        print(f"  Fail-closed mask  : conf {fail_closed_conf}, two-way, hold {FC_HOLD}, "
              f"detector every {seg_skip_n} frames")
    print(f"  Output            : {output_path}")
    print(f"  Enc output        : {enc_output_dir}")
    print(f"  Headless          : {headless}  |  Save video: {save_video}")
    print("=" * 60)

    os.makedirs(enc_output_dir, exist_ok=True)

    # [0] TTP RSA-4096 public key (Tier 3). Tier 1 never generates or holds a
    # keypair itself -- see encryption.py's fetch_ttp_public_key() docstring.
    # A real, reachable Tier 3 server is required for any non-benchmark run:
    # this is a hard error, not a silent local-keypair fallback, because
    # that fallback is exactly the bug this fixes (Tier 1 previously held
    # both the encrypted data AND the key to decrypt it).
    if not benchmark:
        if not ttp_server:
            raise ValueError(
                "ttp_server is required (e.g. 'https://localhost:8843') unless "
                "benchmark=True -- Tier 1 must fetch the real Tier 3 TTP's public "
                "key, it cannot generate its own keypair."
            )
        print(f"\n[0/4] Fetching TTP public key from {ttp_server} ...")
        ttp_public_key = fetch_ttp_public_key(ttp_server, verify_tls=ttp_verify_tls)
        print(f"    TTP public key fetched ({ttp_public_key.key_size}-bit RSA)")
    else:
        print("\n[0/4] Benchmark mode -- skipping TTP public key fetch")
        ttp_public_key = None

    # [1] Detector + RTMPose
    # Single detector for the whole pipeline: YOLO11n (NCNN, box-only, no
    # segmentation) drives BOTH the person boxes fed to RTMPose AND the
    # yolo11n_boxfill anonymizer, when selected. Previously this used
    # rtmlib.Body(), which hardcodes YOLOX-Nano as the detector internally
    # (rtmlib.tools.solution.body.Body.__init__ always wraps `det=` weights
    # in a YOLOX-architecture inference class) -- meaning YOLOX-Nano ran on
    # every frame regardless of --anonymizer, while YOLO11n (when selected)
    # only ever supplied the anonymization mask, not the pose-driving boxes.
    # Two detectors running per frame, only one of which matched what the
    # paper's eval (results/tier1_detection_eval/) actually measures.
    # RTMPose itself is detector-agnostic (rtmlib.tools.pose_estimation.
    # rtmpose.RTMPose.__call__ just wants a plain [x1,y1,x2,y2] box list),
    # so it's constructed standalone here instead of via Body(), and boxes
    # come from YOLO11nBoxBlur.get_mask_and_boxes() at the call site below.
    print("\n[1/4] Loading YOLO11n (NCNN, detection) + RTMPose-T...")
    _y11_pose_ncnn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ncnn_fp32", "yolo11n_ncnn_model")
    person_detector = YOLO11nBoxBlur(model_name=_y11_pose_ncnn_dir, infer_size=320, conf=0.4)
    pose_model = RTMPose(
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        model_input_size=(192, 256),
        backend='onnxruntime',
        device='cpu',
    )

    # [2] MediaPipe FaceLandmarker
    print("\n[2/4] Loading MediaPipe FaceLandmarker...")
    model_path = 'face_landmarker.task'
    if not os.path.exists(model_path):
        print("  Downloading face_landmarker.task ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
            model_path,
        )
    face_mesh = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=FACE_MESH_MIN_CONF,
            min_face_presence_confidence=FACE_MESH_MIN_CONF,
            output_facial_transformation_matrixes=True,
        )
    )

    # [3] EdgeFace embedding model
    if not benchmark:
        print("\n[3/4] Loading EdgeFace-s-gamma-05 embedding model...")
        try:
            embedder = EmbeddingExtractor(EDGEFACE_ONNX_PATH)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Running WITHOUT face embedding.")
            embedder = None
    else:
        print("\n[3/4] Benchmark mode -- skipping EdgeFace embedding model")
        embedder = None

    # [3a] Gender classifier -- same graceful-degradation pattern as the
    # embedder above; runs on the same best_face_crop selected by
    # PersonState.update_best(), so it rides the existing confidence *
    # face_quality best-frame selection rather than tracking its own.
    if not benchmark:
        print("\n[3/4] Loading gender classifier...")
        try:
            gender_classifier = GenderClassifier(GENDER_ONNX_PATH)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            print("  Running WITHOUT gender classification.")
            gender_classifier = None
    else:
        print("\n[3/4] Benchmark mode -- skipping gender classifier")
        gender_classifier = None

    # [3b] Anonymizer backend
    selfie_seg       = None
    mobile_sam       = None
    yolo_seg         = None
    face_canonicalizer = None
    if anonymizer.startswith("selfie_seg"):
        _models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
        _model_file = ("selfie_segmenter_landscape.tflite"
                       if anonymizer == "selfie_seg1"
                       else "selfie_segmenter.tflite")
        _model_path = os.path.join(_models_dir, _model_file)
        print(f"\n[3b] Loading MediaPipe SelfieSegmentation ({_model_file})...")
        selfie_seg = SelfieSegBlur(model_path=_model_path)
        print(f"     Anonymizer: {anonymizer}")
        print(f"\n[3c] Loading FaceCanonicalizer (expression signal)...")
        face_canonicalizer = FaceCanonicalizer(model_path='face_landmarker.task')
    elif anonymizer == "mobilesam":
        _ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "models", "mobile_sam.pt")
        print(f"\n[3b] Loading MobileSAM (ViT-Tiny)...")
        mobile_sam = MobileSAMBlur(checkpoint_path=_ckpt, device="cpu")
        print(f"     Anonymizer: mobilesam")
    elif anonymizer == "yoloseg":
        _y8_ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolov8n-seg.onnx")
        print(f"\n[3b] Loading YOLOv8-seg-nano ONNX (instance segmentation, infer_size={seg_infer_size})...")
        yolo_seg = YOLOSegBlur(model_name=_y8_ckpt, infer_size=seg_infer_size, conf=0.4)
        print(f"     Anonymizer: yoloseg")
    elif anonymizer == "yoloseg11":
        _y11_ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo11n-seg.onnx")
        print(f"\n[3b] Loading YOLO11n-seg ONNX (instance segmentation, infer_size={seg_infer_size})...")
        yolo_seg = YOLOSegBlur(model_name=_y11_ckpt, infer_size=seg_infer_size, conf=0.4)
        print(f"     Anonymizer: yoloseg11")
    elif anonymizer == "yoloseg11int8":
        _y11_int8_ckpt = os.path.join(os.path.dirname(__file__), "..", "..", "models", "yolo11n-seg-int8.onnx")
        print(f"\n[3b] Loading YOLO11n-seg ONNX INT8 (instance segmentation, infer_size={seg_infer_size})...")
        yolo_seg = YOLOSegBlur(model_name=_y11_int8_ckpt, infer_size=seg_infer_size, conf=0.4)
        print(f"     Anonymizer: yoloseg11int8")
    elif anonymizer == "yoloseg11ncnn":
        # NCNN backend, same yolo11n-seg weights, different execution graph --
        # measured ~1.96x faster than ONNX INT8 on real Pi 5 hardware (31.0ms
        # vs 60.8ms/frame segmentation-only, verified via scripts/ncnn_bench.py
        # with explicit mask-found and frame-read correctness checks). On the
        # dev machine (x86) the two backends are roughly at parity -- this
        # gap is ARM-specific (NCNN's hand-tuned NEON/dot-product kernels vs
        # ONNX Runtime's more general MLAS dispatch), consistent with prior
        # per-platform INT8 findings in this file. FP16 export tested
        # pixel-near-identical to FP32 (IoU 0.9998) but no faster on Pi (both
        # ~31ms) -- FP32 used here since there's no speed reason to prefer
        # FP16, and FP32 has zero precision-loss risk.
        #
        # YOLOSegBlur works completely unmodified here: an NCNN model is a
        # directory (models/ncnn_fp32/yolo11n-seg_ncnn_model), not a .onnx
        # file, so its ONNX-specific auto-export/quantize block is a no-op
        # and ultralytics.YOLO() auto-detects the NCNN format from the
        # *_ncnn_model path suffix.
        #
        # NOTE: unlike ONNX (static input shape baked in at export -- a
        # mismatched imgsz raises a clear shape error), passing a
        # seg_infer_size other than the 320 this NCNN model was exported at
        # does NOT error -- it was observed to silently still run (not yet
        # confirmed whether NCNN actually resizes internally to serve a
        # different size correctly, or just ignores the mismatched request
        # and always infers at 320 regardless). Treat seg_infer_size as
        # UNVERIFIED for this anonymizer until checked directly -- don't
        # assume it's honored the way it is for yoloseg11int8.
        _y11_ncnn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ncnn_fp32", "yolo11n-seg_ncnn_model")
        print(f"\n[3b] Loading YOLO11n-seg NCNN (instance segmentation, infer_size={seg_infer_size})...")
        yolo_seg = YOLOSegBlur(model_name=_y11_ncnn_dir, infer_size=seg_infer_size, conf=0.4)
        print(f"     Anonymizer: yoloseg11ncnn")
    elif anonymizer == "yolo11n_boxfill":
        # Plain (detection-only) YOLO11n, NCNN backend -- no segmentation.
        # Reuses the yolo_seg variable slot: YOLO11nBoxBlur deliberately
        # implements the same get_mask_and_boxes(frame) -> (mask, boxes)
        # interface as YOLOSegBlur (see blur_yolo11n.py's class docstring),
        # so every downstream call site below (skip-frame mask-warp
        # propagation, apply_mask, export-mode mask writers) works
        # unmodified -- the "mask" returned is a rasterized rectangle, not
        # a real per-pixel segmentation, but every consumer only ever
        # treats it as an opaque bool array.
        #
        # This is the detector actually measured for the paper's Table 6/7
        # AP/AR results (results/tier1_detection_eval/) -- prior to this
        # wiring it was only ever invoked directly from standalone eval
        # scripts, never through this CLI-driven pipeline.
        # (YOLO11nBoxBlur imported at module level, top of file -- no local
        # import here, a local import anywhere in this function makes the
        # name local to the WHOLE function per Python scoping rules, which
        # broke the earlier person_detector = YOLO11nBoxBlur(...) call above.)
        _y11_box_ncnn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ncnn_fp32", "yolo11n_ncnn_model")
        print(f"\n[3b] Loading YOLO11n NCNN (detection-only, box grey-fill, infer_size={seg_infer_size})...")
        yolo_seg = YOLO11nBoxBlur(model_name=_y11_box_ncnn_dir, infer_size=seg_infer_size,
                                  conf=fail_closed_conf if fail_closed else 0.4)
        print(f"     Anonymizer: yolo11n_boxfill")
        # Same face_canonicalizer wiring as selfie_seg (see that branch above)
        # -- without this, yolo11n_boxfill produced NO face expression signal
        # at all: face_canonicalizer stayed None (only ever set for
        # selfie_seg*), and the separate convexhull-only FaceMesh path
        # (`elif anonymizer == "convexhull"` further down) doesn't match this
        # anonymizer either. That's a real functional gap, not just a timing
        # display issue -- yolo11n_boxfill is the detector actually measured
        # for the paper's results (results/tier1_detection_eval/), and the
        # paper's Tier 1 design requires the 12-dim expression vector as one
        # of the exported signals regardless of which anonymizer produced the
        # grey-fill. Found 2026-08-14 while investigating why the timing
        # summary showed 0.0ms for both FaceMesh and Canonical in this mode.
        print(f"\n[3c] Loading FaceCanonicalizer (expression signal)...")
        face_canonicalizer = FaceCanonicalizer(model_path='face_landmarker.task')
    else:
        print(f"\n[3b] Anonymizer: convexhull")

    # [4] Video IO
    # A bare integer (e.g. "0") means a live camera device index rather than
    # a file path -- cv2.VideoCapture("0") would otherwise try (and fail) to
    # open a file literally named "0". Live sources use CAP_DSHOW, not
    # CAP_MSMF: MSMF hangs indefinitely (not just fails -- genuinely blocks
    # forever on cap.open()) for at least one tested device (Logitech B525),
    # a known MSMF/UVC-driver incompatibility on some older webcams. DSHOW
    # opens instantly for the same device. Confirmed via isolated testing
    # 2026-08-12 -- do not switch back to MSMF without re-verifying against
    # real hardware first.
    is_live_camera = isinstance(input_path, int) or (
        isinstance(input_path, str) and input_path.isdigit()
    )
    print("\n[4/4] Opening video..." if not is_live_camera else "\n[4/4] Opening live camera...")
    # Live capture stays at the device's native 1920x1080 (the only mode
    # confirmed working -- see CAP_DSHOW note above); every frame is then
    # center-cropped to a 1080x1080 square immediately after cap.read(), so
    # width/height/scale/VideoWriter below all see 1080x1080 without needing
    # separate square-aware branches downstream. Matches this project's
    # existing square-clip convention (e.g. 6_single_face.mp4 @ 1264x1264).
    live_crop_x0 = None
    if is_live_camera:
        # CAP_DSHOW is Windows-only (DirectShow) -- doesn't exist as a
        # concept on Linux/the Pi. CAP_V4L2 is the Linux equivalent; picked
        # explicitly (not left to OpenCV's default backend selection) for
        # the same reason CAP_DSHOW was pinned on Windows -- avoid relying
        # on whichever backend happens to be first in ELF/DLL search order.
        _backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(int(input_path), _backend)
        # MJPEG must be requested BEFORE resolution/fps: this webcam (B525,
        # confirmed via v4l2-ctl --list-formats-ext on the Pi) only offers
        # 1920x1080 at 5fps in its raw YUYV mode -- 30fps at that resolution
        # only exists under MJPG (compressed). Without this, V4L2 silently
        # keeps its default format (YUYV) and cap.set(CAP_PROP_FPS, 30)
        # below has no effect at 1920x1080 -- verified capture would
        # silently run at 5fps with no error. FOURCC has no such trap on
        # Windows/DSHOW (which already negotiates MJPEG at 1080p30 for this
        # device) but setting it explicitly there too costs nothing and
        # keeps both platforms' open sequence identical.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"\nError: Could not open '{input_path}'")
        return

    if is_live_camera:
        _native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        _native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        _square   = min(_native_w, _native_h)
        live_crop_x0 = (_native_w - _square) // 2
        live_crop_y0 = (_native_h - _square) // 2
        width, height = _square, _square
        print(f"Live capture native: {_native_w}x{_native_h} -> center-cropped to {width}x{height}")
    else:
        width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS) or 30.0

    scale            = min(width, height) / BASE_RESOLUTION
    FAR_THRESHOLD    = max(int(BASE_FAR_THRESHOLD    * scale), 5)
    MEDIUM_THRESHOLD = max(int(BASE_MEDIUM_THRESHOLD * scale), 15)
    SLOW_THRESHOLD   = max(int(BASE_SLOW_THRESHOLD   * scale), 1)
    FAST_THRESHOLD   = max(int(BASE_FAST_THRESHOLD   * scale), 3)
    kp_scale_x       = width  / INFER_SIZE
    kp_scale_y       = height / INFER_SIZE

    print(f"\nInput  : {width}x{height} @ {fps_input:.1f} fps  (scale={scale:.3f})")
    print(f"Infer  : {INFER_SIZE}x{INFER_SIZE}  kp_scale=({kp_scale_x:.3f}, {kp_scale_y:.3f})")
    print(f"FAR={FAR_THRESHOLD}px  MED={MEDIUM_THRESHOLD}px  "
          f"SLOW={SLOW_THRESHOLD}px  FAST={FAST_THRESHOLD}px\n")

    # ========================================================================
    # MIRAGE re-ID defences -- ONE-TIME SETUP.
    #
    # Everything real is inside `if gait_anon or mask_enabled:`. With both off
    # this block binds a handful of None/0 locals and does nothing else -- no
    # vendored import, no sys.path mutation, no config resolution.
    # ========================================================================
    mask_anon                  = None    # MaskAnonymizer instance, or None
    defence_cfg                = None    # provenance.resolve_config(), computed ONCE
    gait_run_seed              = None    # per-RUN gait seed (never logged, never written)
    _anonymize_export_rows     = None    # bound lazily, only when gait_anon
    emit_mask_small_rows       = None    # downscaled emitted masks, for the containment metric
    mask_superset_violations   = 0       # frames where emit_mask LOST a pre-mitigation pixel
    mask_superset_violation_px = 0
    t_mask_anon_total          = 0.0
    t_gait_anon_total          = 0.0
    _last_gait_prov            = None    # last flush's adapter provenance, for the end-of-run report
    mask_temporal_win_eff      = None
    mask_temporal_win_source   = None
    effective_mask_fps         = None
    _mask_buf_w = _mask_buf_h  = 0
    _mask_buf_sx = _mask_buf_sy = 1.0
    # Whether the EXPORTED confidence column is binarized. Tri-state kwarg:
    # None (default) follows gait_anon, because the raw per-joint confidence
    # trace is itself an identity side-channel and the vendored emit binarizes
    # it alongside the pose (pose_anon_edge.binarize_pose_scores' docstring:
    # "the raw per-joint confidence trace is an identity side-channel").
    # Explicit True/False overrides. NOTHING in the frame loop is affected --
    # see the np.save site for why.
    binarize_enabled = bool(gait_anon) if score_binarize is None else bool(score_binarize)

    if gait_anon or mask_enabled:
        # Lazy on purpose: importing the vendored package inserts its own
        # directory at the FRONT of sys.path (vendor/mirage_edge/__init__.py), which
        # a default run must not pay for or be exposed to.
        from . import provenance as _prov

        # ---- temporal window: a DURATION, resolved against the EMITTED fps --
        # Upstream stores 0.14 s (config.py:512 MASK_TEMPORAL_S) and resolves it
        # to a frame count at EMIT_FPS -- which is why "the same config" is a
        # different defence at a different fps (30->4, 15->2, 10->1).
        #
        # The ring buffer takes ONE entry per EMITTED frame. This pipeline
        # applies AND writes a mask on every frame -- skip frames included, they
        # carry a warped mask forward -- so the emitted-mask rate is the clip's
        # own fps, NOT fps/skip_n and NOT fps/seg_skip_n. Derived, never a
        # constant fitted to a clip.
        MASK_TEMPORAL_S    = 0.14
        effective_mask_fps = float(fps_input)
        # DEFAULT IS NOW A PINNED 2, not the derivation (owner decision 2026-08-14).
        # Measured on this host, p01_c02.mp4, 43 f @30 fps, bbox: frame 42 is a real
        # detection drop, and at win=1 the emitted mask is EMPTY there -- the person is
        # unmasked for that frame. win=2 covers it (81 836 px, 0 empty frames) for
        # +0.98 % median emitted area; the fps-derived 4 costs +1.8 % for no measured
        # gain. 2 is also the window A.6o measured `bbox` at, so source == tested.
        # Pass None or 0 to restore the fps derivation.
        if mask_temporal_win is not None and int(mask_temporal_win) > 0:
            mask_temporal_win_eff    = max(1, int(mask_temporal_win))
            mask_temporal_win_source = "explicit/pinned (mask_temporal_win / --mask-temporal-win)"
        else:
            mask_temporal_win_eff    = max(1, int(round(MASK_TEMPORAL_S * effective_mask_fps)))
            mask_temporal_win_source = (f"derived: max(1, round({MASK_TEMPORAL_S} s * "
                                        f"{effective_mask_fps:.4g} fps))")
            if mask_enabled and mask_temporal_win_eff == 1:
                print(
                    "\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "!! SILHOUETTE DEFENCE: temporal window resolved to 1 -- THE RUNG THAT\n"
                    "!! WAS MEASURED TO GIVE THE LEAST PROTECTION, and no override was given.\n"
                    f"!! {MASK_TEMPORAL_S} s x {effective_mask_fps:.4g} fps rounds to 1 frame, so the running-max\n"
                    "!! over previous masks degenerates to 'this frame only' and the temporal\n"
                    "!! half of the mitigation is inert.\n"
                    "!! On the MIRAGE host this rung (their EMIT_FPS=10) measured 64.53 % NM\n"
                    "!! against 55.03 % for win=2 -- ~9.5 pp LESS containment, and win=2 is\n"
                    "!! the window every 'bbox' number (ledger A.6o) was measured at.\n"
                    "!! Those figures were measured THERE, on THEIR footage; what carries\n"
                    "!! over is the ORDERING, not the values.\n"
                    "!! Pass --mask-temporal-win 2 to pin the measured arm.\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
                    file=sys.stderr, flush=True)

        # ---- ONE resolution of "what is running", passed around from here ---
        # resolve_config() RAISES if it is ever computed twice with a different
        # answer; that is the whole point (two independent computations are how
        # an artifact ended up declaring gait_preset:"" while carrying e2 knobs,
        # ledger A.2d). Nothing below recomputes it -- they read defence_cfg.
        #
        # reset_active() first, because the guard is process-global while a run
        # is one process_video() call: a batch script or an eval sweep calling
        # this function twice at two configs is TWO RUNS, not one config
        # computed twice, and it would otherwise raise on the second clip.
        # (Found by running the wiring twice in one process.) The guard keeps
        # its teeth WITHIN a run -- any second, disagreeing computation after
        # this point still raises against the entry we are about to install.
        _prov.reset_active()
        defence_cfg = _prov.resolve_config(
            gait_preset_name = gait_preset,
            level            = gait_level,
            mask_shape_mode  = mask_shape_mode,
            temporal_win     = mask_temporal_win_eff,
            fps              = float(fps_input),
            gait_enabled     = bool(gait_anon),
            mask_enabled     = bool(mask_enabled),
            mask_fps         = effective_mask_fps,
            binarize_thresh  = float(score_binarize_thresh),
            export_people    = export_people if export_enabled else None,
            # Declared HERE so it is part of the ONE resolution and lands in the
            # digest. resolve_config cannot see it any other way: the pin is
            # applied via MIRAGE_TEST_FIXED_SEED inside gait_anon._PinnedEnv, for
            # the duration of each flush's call only, which is long after this
            # runs. Without it TIER1_CONFIG.json declared `test_artifact: false`
            # with an EMPTY warnings list on a run whose own top-level PRIVACY key
            # said "TEST ARTIFACT - DO NOT SHIP". Only the boolean crosses; the
            # seed value never does.
            deterministic_seed_pinned = bool(gait_anon and gait_pin_run_seed),
            extra = {
                "host":                     "mirage.pipeline.process_video",
                "anonymizer":               anonymizer,
                "seg_skip_n":               int(seg_skip_n),
                **({"fail_closed": {"mask_conf": float(fail_closed_conf), "pose_conf": FC_POSE_CONF,
                                    "hold": FC_HOLD, "two_way": True}} if fail_closed else {}),
                "dense_export":             bool(dense_export),
                "export_enabled":           bool(export_enabled),
                "score_binarize":           bool(binarize_enabled),
                "score_binarize_source":    ("explicit kwarg" if score_binarize is not None
                                             else "follows gait_anon"),
                "mask_temporal_win_source": mask_temporal_win_source,
                "mask_emit_fps_note":       ("one ring-buffer entry per EMITTED frame; this "
                                             "pipeline emits a mask every frame (skip frames "
                                             "carry a warped mask), so emitted-mask fps == "
                                             "clip fps"),
                "gait_on_degenerate":       str(gait_on_degenerate),
                "gait_seed_policy":         ("one new_clip_seed() per RUN, reused on every "
                                             "flush (gait_pin_run_seed=True)" if gait_pin_run_seed
                                             else "vendored per-SEQUENCE draw, re-drawn on "
                                                  "every flush (gait_pin_run_seed=False)"),
            },
        )

        if mask_enabled:
            from .mask_anon import MaskAnonymizer
            # Draws its OWN per-clip seed from the CSPRNG internally and never
            # exposes it. `bbox` does not consume it; every other mode does.
            mask_anon = MaskAnonymizer(shape_mode   = mask_shape_mode,
                                       temporal_win = mask_temporal_win_eff)
            if not blur_bodies:
                # Contradictory request: with anonymization off there is no mask
                # to mitigate, so the defence would sit at 0 frames while the
                # manifest declared it enabled. Not fatal (unlike the backend
                # refusal above, this is a knob the caller can just turn back
                # on), but it must not pass silently -- `mask.runtime.frames`
                # will read 0 and that is what a reader should check.
                print("  WARNING: mask_shape_mode is set but blur_bodies=False (--no-blur). "
                      "No mask is produced, so the silhouette defence will process 0 frames "
                      "even though the manifest declares it enabled. Check "
                      "manifest.json -> mask.runtime.frames.", file=sys.stderr, flush=True)
        if gait_anon:
            from .gait_anon import anonymize_export_rows as _anonymize_export_rows
            if not export_enabled:
                print("  WARNING: gait_anon=True but no --export-dir was given. The gait "
                      "defence transforms the EXPORTED keypoint arrays only -- it does not "
                      "touch the rendered video, the skeleton overlay or the mask. Nothing "
                      "will be anonymised by it in this run.", file=sys.stderr, flush=True)
            if float(score_binarize_thresh) != 0.5:
                # 🔴 ONE KNOB, TWO JOBS. `score_binarize_thresh` is documented as "the
                # threshold at which the EXPORTED confidence column is collapsed to {0,1}",
                # but it is ALSO passed straight down as the gait adapter's `conf_thresh`
                # (see the _anonymize_export_rows call below), which decides which rows count
                # as PRESENT and therefore which rows get transformed at all. gait_anon's own
                # POSE_THRESH is 0.5 because that is the value the VENDORED code hardcodes for
                # its `_unobs` gate (pose_anon_edge.py:2309); any other value desynchronises
                # "present here" from "observed down there".
                # MEASURED on this host, 60 frames of one fully-detected person at a uniform
                # confidence of 0.62: at 0.5 all 60 frames are transformed; at 0.9 the adapter
                # returns `applied: False` and all 60 rows BIT-IDENTICAL, i.e. that person's
                # RAW gait is what reaches keypoints_p*.npy. It is declared (it lands in
                # raw_passthrough_frames / raw_passthrough_breakdown.low_confidence_rows), but
                # nothing about the flag's name or help says the privacy defence rides on it.
                print(f"  WARNING: score_binarize_thresh={float(score_binarize_thresh)} is not "
                      f"0.5. That knob is ALSO the gait defence's presence gate, and the "
                      f"vendored unobserved-joint gate is hardcoded at 0.5. Above 0.5, rows "
                      f"the detector really did find are left UNTRANSFORMED (raw gait on "
                      f"disk); below it, rows the vendored code will zero out are fed in. "
                      f"Check manifest.json -> anon.runtime.raw_passthrough_breakdown."
                      f"low_confidence_rows before trusting this export.",
                      file=sys.stderr, flush=True)

        print("=" * 60)
        print("  MIRAGE re-ID DEFENCES")
        print(f"  Gait (pose)       : {'ON' if gait_anon else 'off'}"
              + (f"  preset={defence_cfg['gait']['preset']!r} level={gait_level} "
                 f"kwargs={defence_cfg['gait']['anonymize_v2_kwargs_n']} "
                 f"on_degenerate={gait_on_degenerate}" if gait_anon else ""))
        print(f"  Silhouette (mask) : {'ON' if mask_enabled else 'off'}"
              + (f"  mode={mask_shape_mode!r} win={mask_temporal_win_eff} "
                 f"({mask_temporal_win_source})" if mask_enabled else ""))
        print(f"  Score binarize    : {'ON' if binarize_enabled else 'off'}"
              + (f"  thresh={float(score_binarize_thresh)} (EXPORTED column only)"
                 if binarize_enabled else ""))
        print(f"  Config digest     : {defence_cfg['digest'][:16]}")
        for _w in defence_cfg["warnings"]:
            print(f"  ! {_w}")
        print("  NOTE: every MIRAGE privacy figure quoted in this code was measured on the")
        print("        MIRAGE edge host, NOT here. None of them describes this pipeline.")
        print("=" * 60)

    # Three-panel output when canonicalizer is active: original | blurred | canonical
    DISPLAY_H  = 640
    _out_w     = (DISPLAY_H * 2 + DISPLAY_H) if face_canonicalizer is not None else width
    _out_h     = DISPLAY_H                   if face_canonicalizer is not None else height

    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps_input, (_out_w, _out_h))
        if not out.isOpened():
            print("  Warning: Could not open VideoWriter -- video will not be saved.")
            out = None

    # --- Dense per-person export setup (opt-in; no-op when export_dir is None) ---
    EXPORT_FACE_SIZE = 512
    slot_tracker              = None
    export_kp_rows            = None
    export_identity_rows      = None   # stable identity id per slot per frame -- see the append sites
    export_bbox_rows          = None
    export_face_param_rows    = None   # parametric, identity-free signal -- the safe-to-transmit face export
    export_valid_smiles       = None   # smile scalar from genuinely-detected (non-held-over) frames only, for the clip baseline
    export_last_face_crop     = None   # transient only: feeds extract_params() + optional diagnostic write, never itself "the" export
    export_last_face_params   = None   # last-good parametric scalars, held across brief absences (mirrors the old face-crop hold-over)
    # stream_id -> (gender_label, gender_conf), populated once per stream at
    # flush_to_disk() time (same single-best-frame classification the normal,
    # non-export path already uses -- see PersonState.flush_to_disk()).
    # Replaces an earlier per-frame-voting-then-averaging scheme that lived
    # only in export mode and disagreed with the non-export path's answer;
    # both paths now derive gender identically, this dict just makes the
    # single result available to the manifest writer after the owning
    # PersonState has been deleted from person_states (which happens on
    # every departure, mid-clip or not -- see the `del person_states[dep_id]`
    # a few lines below where flush_to_disk() is called).
    stream_gender_by_id       = {}
    export_slot_stream_id     = None
    export_face_canon         = None
    export_face_writers       = None   # diagnostics-only now (see export_diagnostics gating below)
    export_rtm_writer         = None
    export_mask_writer        = None
    export_raw_mask_writer    = None
    export_gate_writer        = None
    export_bbox_overlay_writer = None
    export_clip_id            = None
    export_crypto_dir         = None

    if export_enabled:
        from .export_tracking import ExportSlotTracker
        slot_tracker            = ExportSlotTracker(export_people, width, height)
        export_kp_rows          = [[] for _ in range(export_people)]
        # Parallel to export_kp_rows: WHO was in that slot on that frame
        # (identity_tracker's stable id), or None. Export slots are RE-LET
        # after repeated misses, so without this one slot block can splice two
        # different people into one sequence -- handing them a shared collapse
        # template and a shared seed, which is the linkable pseudo-identity the
        # per-sequence draw exists to prevent. Buffered unconditionally (one
        # int per slot per frame) so identities_p{i}.npy is always available
        # for a later offline transform, defence on or off.
        export_identity_rows    = [[] for _ in range(export_people)]
        export_bbox_rows        = [[] for _ in range(export_people)]
        export_face_param_rows  = [[] for _ in range(export_people)]
        export_valid_smiles     = [[] for _ in range(export_people)]
        export_last_face_crop   = [None] * export_people
        export_last_face_params = [None] * export_people
        # Filled in per-frame from the real PersonState.stream_id occupying
        # each slot (see the slot_matches loop below) -- NOT a fresh uuid4
        # minted here. Export slots and PersonState streams are otherwise
        # two unrelated id spaces (slots are stable left-to-right identities
        # for the whole clip; PersonState streams churn on identity loss),
        # so this is the one place they get tied together, letting the
        # phone locate the right .packet/.key crypto files for a given slot.
        export_slot_stream_id   = [None] * export_people
        export_face_canon       = FaceCanonicalizerV2(model_path='face_landmarker.task')
        export_clip_id          = str(uuid.uuid4())
        # PersonState's crypto output (.packet/.key, see tracking.py) is
        # redirected here instead of enc_output_dir, so it lands inside the
        # same directory tree tier1_link/server.py already serves to the
        # phone -- that server just walks every file under a clip dir, no
        # server-side change needed for it to pick these up.
        export_crypto_dir       = os.path.join(export_dir, "crypto")
        os.makedirs(export_crypto_dir, exist_ok=True)

        _exp_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        export_rtm_writer  = cv2.VideoWriter(os.path.join(export_dir, "output_rtm.mp4"),
                                              _exp_fourcc, fps_input, (width, height))
        export_mask_writer = cv2.VideoWriter(os.path.join(export_dir, "mask.mp4"),
                                              _exp_fourcc, fps_input, (width, height), isColor=False)
        if export_diagnostics:
            # face_crops_p{i}.mp4 is a RAW, unblurred face crop -- useful for local
            # debugging (visually checking what face_params_p{i}.npy was derived
            # from) but it must never be treated as part of the safe-to-transmit
            # bundle. Gated behind export_diagnostics for exactly that reason,
            # same as the other raw/debug-only outputs below.
            export_face_writers = [
                cv2.VideoWriter(os.path.join(export_dir, f"face_crops_p{i}.mp4"),
                                 _exp_fourcc, fps_input, (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE))
                for i in range(export_people)
            ]
            export_raw_mask_writer = cv2.VideoWriter(
                os.path.join(export_dir, "raw_seg_mask.mp4"),
                _exp_fourcc, fps_input, (width, height), isColor=False)
            export_gate_writer = cv2.VideoWriter(
                os.path.join(export_dir, "gate_region.mp4"),
                _exp_fourcc, fps_input, (width, height), isColor=False)
            export_bbox_overlay_writer = cv2.VideoWriter(
                os.path.join(export_dir, "bbox_overlay.mp4"),
                _exp_fourcc, fps_input, (width, height))
        print(f"\n  Dense export enabled -> {export_dir}  "
              f"(slots={export_people}, diagnostics={export_diagnostics})")

        # ---- ONE gait seed for the whole run --------------------------------
        # THE PROBLEM IT ADDRESSES: _write_export_arrays() re-transforms the
        # ENTIRE buffer on every flush (every EXPORT_FLUSH_EVERY frames), so a
        # clip that flushes N times puts N different views of the same
        # underlying gait through keypoints_p{i}.npy. Anyone with read access to
        # export_dir during the run collects them for free, and averaging
        # independent draws attenuates the perturbation.
        #
        # 🔴 WHAT PINNING ACTUALLY BUYS: NOTHING MEASURABLE. I first wrote here
        #    that pinning makes every flush emit the IDENTICAL transform. THAT
        #    IS FALSE -- pinning freezes the RANDOM draw, but the collapse
        #    template is re-fit over a longer buffer at each flush, so the views
        #    still differ. Measured on a 70-frame 2-slot synthetic clip
        #    (3 flushes at 30/60/70), max |delta| over the 30-frame overlap
        #    between flush 1 and flush 2, n=8 runs per arm:
        #        pinned        23.15 +/- 7.63 px   range [14.28, 40.06]
        #        per-sequence  25.31 +/- 4.26 px   range [19.31, 31.90]
        #        -> 8.5 % lower on the mean, lower in 6/8 paired runs, i.e.
        #           INDISTINGUISHABLE FROM NOISE.
        #    A single draw first showed a 39 % reduction; at n=8 that was noise.
        #    So the averaging channel is NOT closed and is not even measurably
        #    narrowed. The justification for pinning does not survive its own
        #    measurement.
        #
        # 🔴 AND THE COST IS REAL. The vendored code consumes a pinned seed via
        #    MIRAGE_TEST_FIXED_SEED, which makes it deterministic AND identical
        #    across every sequence in the call -- so every tracklet in the clip
        #    shares one perturbation, collapsing the per-SEQUENCE draw that
        #    the internal runbook records as load-bearing (per-identity seeding leaked
        #    6-8x chance on every measured arm). The seed is still fresh per RUN
        #    (secrets CSPRNG, never derived from identity or content), so nothing
        #    links ACROSS clips. Because MIRAGE_TEST_FIXED_SEED is honoured, the
        #    manifest is stamped "PRIVACY: TEST ARTIFACT - DO NOT SHIP" whenever
        #    this is on -- i.e. the default configuration cannot produce a
        #    shippable bundle. That is deliberate and visible, not a bug.
        #
        # ✅ RESOLVED 2026-08-14 - THE DEFAULT IS NOW False. The integration brief
        #    specified True; the measurement above landed afterwards and refuted
        #    its premise, so the brief's author was asked and flipped it. False
        #    is the vendored per-SEQUENCE draw, i.e. the arm MIRAGE actually
        #    measured, and it costs nothing measurable (8.5 % spread reduction at
        #    n=8, within noise). A SECOND reason arrived with the smoke run: with
        #    pinning on, two identical ON runs still differed by max 74.63 px
        #    because new_clip_seed() is secrets.randbits(31) when
        #    MIRAGE_TEST_FIXED_SEED is unset -- so pinning did not even deliver
        #    the reproducibility the manifest claimed for it.
        #    The genuinely correct fix is still neither setting: stop
        #    re-transforming on interim flushes at all -- buffer raw rows to a
        #    scratch path and transform once at clip end, or make the flush
        #    append-only. That is a change to the flush design, not to a seed,
        #    and it remains OPEN.
        #
        # The value is held in a local, is never printed, and never reaches any
        # file -- gait_anon's provenance reports only the boolean.
        if gait_anon and gait_pin_run_seed:
            from .vendor.mirage_edge import new_clip_seed as _new_clip_seed
            gait_run_seed = int(_new_clip_seed())
            # ASCII on purpose. MEASURED on this box: sys.stdout is cp1252 with
            # errors='surrogateescape' and sys.stderr is cp1252 with
            # errors='backslashreplace'. So a non-latin-1 char in a printed string
            # either renders as a literal backslash escape (stderr) or RAISES
            # UnicodeEncodeError and kills the process (stdout -- which is how
            # argparse writes --help). A privacy warning that reads as mojibake is
            # a warning people skip. Comments and docstrings in this file are free
            # to use whatever they like; printed strings are not.
            print("  !! gait_pin_run_seed=True: one fixed seed for this whole run, so the "
                  "perturbation is deterministic and SHARED by every tracklet in the clip. "
                  "The export will be stamped 'PRIVACY: TEST ARTIFACT - DO NOT SHIP'. "
                  "Pass gait_pin_run_seed=False for the per-sequence draw.",
                  file=sys.stderr, flush=True)

    # Downscaled copies of the EMITTED mask, one per frame, kept only when a
    # defence is running and there is an export to report into. ~256 px max dim
    # (65 kB/frame at most) so a whole clip stays trivially in memory, and the
    # containment metric below is measured in that same space.
    if export_enabled and (gait_anon or mask_enabled):
        _s           = min(1.0, MASK_BUF_MAX_DIM / float(max(width, height)))
        _mask_buf_w  = max(1, int(round(width  * _s)))
        _mask_buf_h  = max(1, int(round(height * _s)))
        _mask_buf_sx = _mask_buf_w / float(width)
        _mask_buf_sy = _mask_buf_h / float(height)
        emit_mask_small_rows = []

    def _write_export_arrays():
        """Re-writes keypoints_p{i}.npy / bboxes_p{i}.json / face_params_p{i}.npy
        / manifest.json from the current in-memory export_* rows. Called
        periodically during the loop (see EXPORT_FLUSH_EVERY below) AND once
        more at clip-end -- so a run interrupted mid-clip (e.g. a live-camera
        session with no natural EOF, see is_live_camera) still leaves a valid,
        current-as-of-last-flush export on disk instead of losing everything,
        at the cost of periodically re-writing the whole array (cheap at the
        frame counts this pipeline runs -- these aren't multi-hour clips).

        The MIRAGE gait defence runs HERE, on EVERY flush -- not once at clip
        end. It has to: this function writes keypoints_p{i}.npy every
        EXPORT_FLUSH_EVERY frames, so a transform that only ran at the end
        would leave RAW gait on disk for the whole run and only overwrite it
        afterwards."""
        nonlocal t_gait_anon_total, _last_gait_prov

        # ---- MIRAGE gait defence (opt-in) ----------------------------------
        # `kp_src` IS `export_kp_rows` when the defence is off, so the default
        # path below is the same computation on the same objects it always was.
        #
        # When it is on, the adapter returns NEW rows and leaves the raw buffer
        # untouched -- which is required, not incidental: this function re-runs
        # on the next flush and assigning the result back would double-transform.
        kp_src, gait_prov = export_kp_rows, None
        if gait_anon:
            # Fact, not belief: export mode forces dense_export (:104), which
            # forces SKIP_N all-1 (:123), so every buffered row is one real
            # frame and the transform's fps is the clip's own. If that ever
            # stops holding, the rows are a subsampled series and every
            # seconds-domain knob in the transform is silently wrong -- so this
            # fails loudly instead of emitting a plausible-looking result.
            if current_N != 1:
                raise RuntimeError(
                    f"gait_anon requires one buffered row per real frame, but "
                    f"current_N == {current_N} (skip-frame propagation is active). "
                    f"Export mode is supposed to force dense_export=True -> "
                    f"SKIP_N all-1; something changed that. Refusing to feed a "
                    f"subsampled series to a transform whose cadence/phase knobs "
                    f"are all expressed in SECONDS.")
            _tg0 = time.time()
            kp_src, gait_prov = _anonymize_export_rows(
                export_kp_rows, export_identity_rows, float(fps_input),
                preset         = gait_preset,
                level          = gait_level,
                seed           = gait_run_seed,        # None unless pinned per run
                frame_wh       = (width, height),
                conf_thresh    = float(score_binarize_thresh),
                on_degenerate  = gait_on_degenerate,
            )
            t_gait_anon_total += time.time() - _tg0

            # 🔴 THE CONFIG IS RESOLVED IN TWO PLACES, SO CROSS-CHECK IT HERE.
            # provenance.resolve_config() merges LEVELS[level] with the preset
            # after popping the wrapper-only keys; gait_anon.anonymize_export_rows
            # does the SAME merge independently, against its own copy of the
            # popped-key list (_HOST_LEVEL_PRESET_KEYS vs _POPPED_WRAPPER_KEYS).
            # Both land in manifest.json -- as `anon.anonymize_v2_kwargs` and
            # `anon.runtime.anonymize_v2_kwargs` -- and nothing compared them, so
            # the two lists could drift and the artifact would carry two different
            # answers to "what ran" with no complaint. That is §A.2d exactly: one
            # config declared, another carried. They agree today (12 keys, checked
            # for e2/L4); this makes the day they stop a hard failure, not a
            # discrepancy someone notices in a JSON later.
            _cfg_kwargs = defence_cfg["gait"]["anonymize_v2_kwargs"]
            _run_kwargs = gait_prov.get("anonymize_v2_kwargs")
            if _run_kwargs != _cfg_kwargs:
                _only_cfg = {k: _cfg_kwargs[k] for k in set(_cfg_kwargs) - set(_run_kwargs)}
                _only_run = {k: _run_kwargs[k] for k in set(_run_kwargs) - set(_cfg_kwargs)}
                _diff = {k: (_cfg_kwargs[k], _run_kwargs[k])
                         for k in set(_cfg_kwargs) & set(_run_kwargs)
                         if _cfg_kwargs[k] != _run_kwargs[k]}
                raise RuntimeError(
                    "the resolved gait config and the config the adapter actually "
                    "ran DISAGREE -- refusing to write an artifact that declares "
                    "one and carries the other (ledger §A.2d).\n"
                    f"  only in provenance.resolve_config: {_only_cfg}\n"
                    f"  only in gait_anon adapter        : {_only_run}\n"
                    f"  differing values                 : {_diff}\n"
                    "  Most likely cause: provenance._POPPED_WRAPPER_KEYS and "
                    "gait_anon._HOST_LEVEL_PRESET_KEYS have drifted apart.")
            _last_gait_prov = gait_prov

        manifest_slots = []
        last_total_frames = 0
        for i in range(export_people):
            kp_arr = (np.stack(kp_src[i], axis=0) if kp_src[i]
                      else np.zeros((0, 17, 3), dtype=np.float32))
            if binarize_enabled and kp_arr.size:
                # ONLY the EXPORTED confidence column, and only here. np.stack
                # already produced a fresh array, so this cannot reach back into
                # export_kp_rows. The in-loop consumers that threshold at 0.3
                # (bboxes_from_keypoints, draw_skeleton, compute_frame_confidence)
                # read `scores` straight off the pose model and are untouched --
                # binarizing what they see would change detection/tracking
                # behaviour, which is not what this is for.
                # Same rule as the vendored binarize_pose_scores (:2436),
                # `1 if s >= thresh else 0`; its own signature takes the MIRAGE
                # emit format ([frames][persons] dicts), not an (N, 17, 3) array,
                # so it does not fit here and the rule is applied directly.
                kp_arr[:, :, 2] = (kp_arr[:, :, 2] >= float(score_binarize_thresh)
                                   ).astype(kp_arr.dtype)
            np.save(os.path.join(export_dir, f"keypoints_p{i}.npy"), kp_arr)
            if defence_cfg is not None:
                # WHO was in this slot, per frame. int32 with -1 for "empty slot or
                # identity lookup missed", so np.load() needs no allow_pickle.
                #
                # 🔴 WRITTEN ONLY WHEN A DEFENCE IS RUNNING. It was unconditional
                # until 2026-08-14, which broke the one hard requirement of this
                # integration: a defences-OFF run must put exactly the files on disk
                # that it put there before the defences existed. MEASURED: HEAD vs
                # working tree over `input/reid_dataset/p01_c02.mp4` --  every one of
                # the 12 shared artifacts was sha256-identical, and the ONLY
                # difference was these three extra files. It is also an artifact
                # worth not emitting by default: `identities_p{i}.npy` states, per
                # frame, WHICH tracked person occupied a slot, so it re-links a
                # re-let slot's two occupants for any reader of the bundle -- a
                # linkage the raw export does not otherwise publish. The BUFFER is
                # still filled unconditionally (one int per slot per frame, in
                # memory); only the write is gated, so the gait adapter still gets
                # its per-identity split with no behavioural difference.
                np.save(os.path.join(export_dir, f"identities_p{i}.npy"),
                        np.array([-1 if v is None else int(v)
                                  for v in export_identity_rows[i]], dtype=np.int32))
            with open(os.path.join(export_dir, f"bboxes_p{i}.json"), "w") as f:
                json.dump(export_bbox_rows[i], f)

            fp_arr = (np.stack(export_face_param_rows[i], axis=0) if export_face_param_rows[i]
                      else np.zeros((0, 12), dtype=np.float32))
            np.save(os.path.join(export_dir, f"face_params_p{i}.npy"), fp_arr)
            last_total_frames = kp_arr.shape[0]

            # Smile baseline uses only genuinely-detected frames (export_valid_smiles),
            # never the held-over values in export_face_param_rows -- matches the
            # two-pass approach in scripts/test_face_canon_v2.py. Downstream
            # rendering applies this correction itself (FaceCanonicalizerV2.
            # set_smile_baseline() + render()) -- exported params are raw/uncorrected.
            smile_baseline = (float(np.median(export_valid_smiles[i]))
                               if export_valid_smiles[i] else 0.0)

            # Gender: single-best-frame classification from stream_gender_by_id,
            # populated once per stream at PersonState.flush_to_disk() time --
            # the exact same value the normal (non-export) path produces for
            # this stream, not a separate export-only computation. None if
            # this slot's stream hasn't flushed yet (still mid-clip -- same
            # timing caveat as packet_file/key_file below) or never had a
            # classifiable face.
            gender_label, gender_conf = stream_gender_by_id.get(
                export_slot_stream_id[i], (None, None)
            )

            # packet_file/key_file: the real crypto bundle (see tracking.py's
            # PersonState.flush_to_disk / __init__) for whichever stream this
            # slot ended up bridged to (see the slot_matches loop above).
            # null if this slot never had a real occupant, or its stream
            # hadn't flushed yet (e.g. still mid-clip -- flush_to_disk() only
            # runs when a person departs or the clip ends -- or crypto
            # disabled via benchmark=True) -- checked by real file existence,
            # not assumed, since "a stream_id was recorded" doesn't guarantee
            # the files were actually written yet.
            packet_file, key_file = None, None
            sid = export_slot_stream_id[i]
            if sid is not None:
                candidate_packet = os.path.join(export_crypto_dir, f"stream_{sid}.packet")
                candidate_key    = os.path.join(export_crypto_dir, f"stream_{sid}.key")
                if os.path.isfile(candidate_packet):
                    packet_file = f"crypto/stream_{sid}.packet"
                if os.path.isfile(candidate_key):
                    key_file = f"crypto/stream_{sid}.key"

            manifest_slots.append({
                "slot": i,
                "stream_id": export_slot_stream_id[i],
                "face_smile_baseline": smile_baseline,
                "frames_with_face": len(export_valid_smiles[i]),
                "gender": gender_label,
                "gender_confidence": gender_conf,
                "packet_file": packet_file,
                "key_file": key_file,
            })
            if defence_cfg is not None:
                # Added as a separate statement so the slot record above stays
                # byte-for-byte what it was when no defence is running.
                manifest_slots[-1]["stick_ink_containment"] = (
                    _stick_ink_containment(kp_arr, emit_mask_small_rows,
                                           _mask_buf_sx, _mask_buf_sy,
                                           float(score_binarize_thresh))
                    if emit_mask_small_rows else None)

        manifest = {
            "clip_id": export_clip_id,
            "fps": fps_input,
            "width": width,
            "height": height,
            "num_slots": export_people,
            "total_frames": last_total_frames,
            "slots": manifest_slots,
        }
        if defence_cfg is not None:
            # ---- what actually ran, from the ONE resolved config ------------
            # `anon`/`mask` = the resolved configuration (what was asked for);
            # their `runtime` sub-blocks = what the run then measured. Both are
            # present only when a defence is on, so a default run's manifest is
            # unchanged, byte for byte.
            test_artifact = bool(defence_cfg.get("test_artifact")) or gait_run_seed is not None
            manifest["anon"] = dict(defence_cfg["gait"], runtime=gait_prov)
            manifest["mask"] = dict(
                defence_cfg["mask"],
                runtime = (dict(mask_anon.stats(),
                                # Measured by the HOST at the seam, independently of
                                # the adapter's own counter: the emitted mask must
                                # never LOSE a pixel the pre-mitigation mask had.
                                # 0 == clean; null == the mask defence was off.
                                host_superset_violation_frames = mask_superset_violations,
                                host_superset_violation_px     = mask_superset_violation_px)
                           if mask_anon is not None else None),
                superset_violations = (mask_superset_violations
                                       if mask_anon is not None else None),
            )
            if test_artifact:
                manifest["PRIVACY"] = "TEST ARTIFACT - DO NOT SHIP"
                manifest["PRIVACY_REASON"] = (
                    "a fixed gait seed is in force (MIRAGE_TEST_FIXED_SEED honoured"
                    + (" via gait_pin_run_seed=True" if gait_run_seed is not None else "")
                    + "), so the perturbation is deterministic and shared across every "
                      "sequence in this clip. That is a linkable pseudo-identity and is "
                      "not privacy-safe to ship.")

        if fail_closed_mask is not None:
            manifest["fail_closed"] = dict(fail_closed_mask.stats(), mask_conf=float(fail_closed_conf))

        with open(os.path.join(export_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if defence_cfg is not None:
            # The full resolved config, beside the manifest, so a bundle can name
            # the exact code (sha256 per vendored file) and knobs that produced it
            # without the reader having to dig it out of the manifest.
            with open(os.path.join(export_dir, "TIER1_CONFIG.json"), "w") as f:
                json.dump({
                    "clip_id": export_clip_id,
                    "generated_by": "mirage.pipeline.process_video",
                    "PRIVACY": manifest.get("PRIVACY"),
                    "note": ("Every MIRAGE privacy figure referenced by this code was "
                             "measured on the MIRAGE edge host, not on this pipeline. "
                             "This file names code and configuration only."),
                    "config": defence_cfg,
                }, f, indent=2)

    EXPORT_FLUSH_EVERY = 30  # ~1s of clip at 30fps -- bounds data loss on interruption without significant periodic I/O overhead

    frame_idx        = 0
    full_frame_count = 0
    skip_frame_count = 0
    person_states    = {}   # identity_id (PersonIdentityTracker's, stable across frames) -> PersonState
    # Nearest-centroid identity tracking for person_states -- fixes a real
    # bug where raw per-frame detection index was treated as a stable
    # identity (it isn't: rtmlib's det/pose order can reshuffle frame to
    # frame). Confirmed causing two different encrypted streams in a
    # 3-person clip to both end up with their best-confidence face crop
    # pulled from the SAME middle physical person -- whichever raw index
    # they happened to occupy on their own best-confidence frame. See
    # export_tracking.py's PersonIdentityTracker docstring.
    identity_tracker = PersonIdentityTracker(width, height)
    current_N        = SKIP_N["medium"]
    movement_tier    = "medium"
    last_keypoints        = None
    last_scores           = None
    last_det_idx_to_identity = {}  # det_idx (as of last full frame) -> identity_id,
                                    # held over skip frames the same way last_keypoints
                                    # is -- convexhull mode's skip-frame face-mesh LK
                                    # path (below) looks up person_states by the raw
                                    # index it tracked last full frame, which is only
                                    # meaningful via this mapping now that person_states
                                    # is keyed by stable identity_id, not raw index.
    last_bboxes           = None
    last_scaled_bboxes    = None   # detector bboxes scaled to frame resolution for MobileSAM
    last_seg_mask         = None   # last selfie-seg mask (bool H×W), propagated on skip frames
    seg_mask_keypoints    = None   # keypoints at the time last_seg_mask was computed/warped
    last_gate_region      = None   # last non-empty bbox_region_mask, held over brief det+pose dropout
    gate_region_stale_for = 0      # consecutive full frames since last_gate_region was refreshed
    last_mesh_cache       = {}
    last_canonical_face   = None   # last canonical expression image (CANONICAL_SIZE×CANONICAL_SIZE)
    last_yolo_boxes_full_space = None  # yolo_seg's own boxes, held over its skip frames
    prev_gray        = None

    prev_time   = time.time()
    fps_history = []
    FPS_WINDOW  = 30

    t_det_total      = 0.0
    t_pose_total     = 0.0
    t_facemesh_total = 0.0
    t_of_body_total  = 0.0
    t_of_face_total  = 0.0
    t_seg_total      = 0.0   # selfie-seg inference (runs parallel to det+pose)
    t_canonical_total = 0.0  # face canonicalizer (every frame)
    t_blur_total     = 0.0   # mask apply + warp only
    t_draw_total     = 0.0
    t_write_total    = 0.0
    t_encrypt_total  = 0.0
    t_embed_total    = 0.0

    # Thread pools for parallelism (all C++ backends release the GIL)
    _seg_pool   = ThreadPoolExecutor(max_workers=1)  # selfie seg ∥ det+pose
    _lk_pool    = ThreadPoolExecutor(max_workers=2)  # body LK ∥ face LK
    # Main-output VideoWriter is now a real persistent write thread (see
    # _write_q / _write_thread below), not a ThreadPoolExecutor.

    streams_flushed = 0
    loop_start      = time.time()

    # Cross-frame pipelining of yolo_seg (running yolo_seg(N+1) in a
    # background thread while pose/draw/write for frame N happen) was tried
    # and measured WORSE on the real pipeline (4.29 FPS vs 5.32 FPS serial),
    # despite a +19% win in an isolated spike (scripts/seg_pipeline_spike.py,
    # which only had yolo_seg + pose_model in flight together). Root cause:
    # ONNX Runtime's default intra_op_num_threads=0 means every ORT session
    # in the process (yolo_seg's, RTMPose-T's det/pose) independently claims
    # the whole physical-core pool for itself -- running two ORT sessions
    # "concurrently" via Python threads doesn't split cores between them, it
    # oversubscribes them, causing contention. The full pipeline also runs
    # MediaPipe's FaceLandmarker (its own thread-hungry backend) on the main
    # thread at the same time, which the isolated spike didn't have, making
    # the real contention worse than the spike predicted. A real fix would
    # need to cap yolo_seg's ORT session to fewer threads, but ultralytics.
    # YOLO()'s plain-.onnx load path takes no SessionOptions param at all
    # (confirmed: AutoBackend calls onnxruntime.InferenceSession(w,
    # providers=providers) with no options arg) -- doing that cleanly means
    # bypassing ultralytics' ONNX loading entirely, out of scope for this
    # change. Kept serial for now.
    # Three-stage read / process / write pipeline, matching MIRAGE's own
    # Tier-1 implementation (a read thread loads frames into a queue, a
    # processing thread handles them and passes results to a write queue,
    # a write thread stores them). The write thread is defined here (before
    # the read thread) so both queues/threads exist before the main
    # processing loop starts submitting to either.
    #
    # Unlike the read queue, the write queue is UNBOUNDED and NEVER drops
    # work for either live or file input: a dropped read frame just means
    # the pipeline worked on slightly-stale-but-still-valid input, but a
    # dropped WRITE means a frame silently goes missing from the output
    # video -- there is no equivalent "it's fine, we'll get the next one"
    # for output correctness. If the write thread ever falls behind
    # processing, the queue grows rather than losing frames; it drains
    # fully on shutdown (queue.join() below) before the VideoWriter closes.
    _write_q = queue.Queue()  # unbounded: (write_fn, frame) tuples, or None sentinel

    def _write_worker():
        while True:
            item = _write_q.get()
            if item is None:
                _write_q.task_done()
                return
            write_fn, payload = item
            write_fn(payload)
            _write_q.task_done()

    _write_thread = threading.Thread(target=_write_worker, daemon=True, name="output-write")
    _write_thread.start()

    # Dedicated read thread: cap.read() must never block on inference, so it
    # runs on its own thread and hands frames to the main (processing) thread
    # via a queue -- together with the write thread above, this is the full
    # 3-stage read / process / write decoupling.
    #
    # Live camera vs. file input need OPPOSITE queue policies:
    #  - Live camera: maxsize=1, and the reader DROPS the previous unread
    #     frame rather than blocking, always keeping only the newest frame
    #     available. Processing a backlog of stale camera frames only adds
    #     latency -- for a live feed you want "now", not "eventually all of
    #     them". This intentionally allows frame loss under load, exactly
    #     like a real camera pipeline would in practice.
    #  - File input: unbounded queue, nothing ever dropped. Every frame in
    #     the file must still be processed for existing benchmark/export
    #     correctness (frame counts, CSV timing, dense export) to hold.
    _frame_q = queue.Queue(maxsize=1 if is_live_camera else 0)
    _read_stop = threading.Event()

    def _read_worker():
        while not _read_stop.is_set():
            ok, raw = cap.read()
            if not ok:
                _frame_q.put(None)  # sentinel: end of stream
                return
            if live_crop_x0 is not None:
                raw = raw[live_crop_y0:live_crop_y0 + height, live_crop_x0:live_crop_x0 + width]
            if is_live_camera:
                # Drop the stale frame (if any) rather than block -- keeps
                # the processing thread on the most recent camera frame.
                try:
                    _frame_q.get_nowait()
                except queue.Empty:
                    pass
            _frame_q.put(raw)

    _read_thread = threading.Thread(target=_read_worker, daemon=True, name="capture-read")
    _read_thread.start()

    # The fail-closed mask needs the next detection before it can settle the
    # frames in front of it, so it reads ahead of the processing loop and hands
    # each frame over together with its finished mask.
    fail_closed_mask = None
    if fail_closed:
        fail_closed_mask = FailClosedMask(yolo_seg.get_boxes_and_conf, _frame_q.get,
                                          width, height, seg_skip_n)

    while cap.isOpened():
        if fail_closed_mask is not None:
            _fc_item = fail_closed_mask.next()
            if _fc_item is None:
                break
            frame, _fc_mask, _fc_boxes = _fc_item
        else:
            frame = _frame_q.get()
            if frame is None:
                break

        annotated     = frame.copy()
        curr_gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_full_frame = (frame_idx % current_N == 0)

        # yolo_seg has its OWN cadence (seg_skip_n), independent of skip_n
        # (which governs det/pose/face-canon) -- default seg_skip_n=1 means
        # every frame, matching the original behavior/reasoning below. This
        # exists as a separate knob because det/pose/face-canon tolerate
        # LK-optical-flow tracking fine, but segmentation mask quality
        # degrades under skip-n's affine-mask-warp (breaks down for non-rigid
        # motion, compounds error across the skip window) -- seg_skip_n=1
        # avoids that entirely at full per-frame cost; seg_skip_n>1 trades
        # some of that quality back for FPS, using the SAME affine-warp
        # mechanism skip_n already uses for pose (see the mask-apply block
        # below), just on its own cadence.
        #
        # BEFORE det+pose so ONNX Runtime CPU threads are idle -- PyTorch and
        # OnnxRuntime both use all CPU threads; running concurrently (or even
        # back-to-back while ORT threads linger) causes severe slowdown.
        #
        # Its internal detect head already finds person boxes as part of
        # get_mask_and_boxes() -- confirmed via direct timing that running
        # the pipeline's separate person_detector (YOLO11n) on top of that
        # was pure duplicated detection work (two independent detectors
        # disagreeing near their own confidence thresholds, ~18ms wasted per
        # full frame for no accuracy benefit; this was measured back when
        # the separate detector was rtmlib's YOLOX-Nano, since replaced by
        # YOLO11n -- see [1] above -- but the same duplication logic applies
        # regardless of which model person_detector wraps). So when yolo_seg
        # is on, person_detector's own call is skipped on full frames and
        # yolo_seg's own boxes drive RTMPose-T's pose_model() instead --
        # EXCEPT on a frame where segmentation itself
        # is being skipped (seg_skip_n>1): there yolo_seg produces no fresh
        # boxes, so pose falls back to keypoint-derived boxes instead (same
        # bboxes_from_keypoints() pattern MobileSAM already uses for its own
        # skip frames) rather than forcing a seg run seg_skip_n was meant to
        # avoid.
        is_seg_full_frame = (frame_idx % seg_skip_n == 0)
        yolo_boxes_full_space = None
        if yolo_seg is not None and blur_bodies:
            if fail_closed_mask is not None:
                # Already settled for this frame: nothing to detect or warp here.
                # seg_mask_keypoints stays None so the skip-frame warp below is off.
                last_seg_mask, yolo_boxes_full_space = _fc_mask, _fc_boxes
                seg_mask_keypoints = None
            elif is_seg_full_frame or prev_gray is None:
                _t_yolo0 = time.time()
                last_seg_mask, yolo_boxes_full_space = yolo_seg.get_mask_and_boxes(frame)
                t_seg_total += time.time() - _t_yolo0
                last_yolo_boxes_full_space = yolo_boxes_full_space
                seg_mask_keypoints = None  # fresh mask this frame -- no warp needed/valid
            else:
                # Seg-skip frame: warp the last real mask forward, same
                # affine-from-keypoint-motion mechanism as skip_n's own skip
                # frames (see the mask-apply block below) -- just triggered
                # on seg_skip_n's cadence instead of skip_n's.
                if last_seg_mask is not None and seg_mask_keypoints is not None \
                        and last_keypoints is not None and len(last_keypoints) > 0:
                    old_pts = seg_mask_keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    new_pts = last_keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    if len(old_pts) == len(new_pts) and len(old_pts) >= 3:
                        M, _ = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC)
                        if M is not None:
                            last_seg_mask = cv2.warpAffine(
                                last_seg_mask.astype(np.uint8), M, (width, height),
                                flags=cv2.INTER_NEAREST
                            ).astype(bool)
                seg_mask_keypoints = last_keypoints.copy() if last_keypoints is not None and len(last_keypoints) > 0 else None
                # No fresh boxes this frame -- fall back to keypoint-derived
                # boxes for pose (only matters if this also happens to be a
                # det/pose full frame; see the is_full_frame block below).
                yolo_boxes_full_space = None

        if is_full_frame or prev_gray is None:
            full_frame_count += 1
            infer_frame = cv2.resize(frame, (INFER_SIZE, INFER_SIZE))

            # selfie_seg: thread pool (TFLite releases GIL → true parallel with det+pose)
            if selfie_seg is not None and blur_bodies:
                _future_seg = _seg_pool.submit(selfie_seg.get_mask, frame, 256)
            else:
                _future_seg = None
            _t_seg0 = time.time()

            if yolo_seg is not None and blur_bodies:
                if yolo_boxes_full_space is not None and len(yolo_boxes_full_space) > 0:
                    # Reuse yolo_seg's own boxes (full-frame space) -> infer_frame space.
                    bboxes = yolo_boxes_full_space.copy().astype(float)
                    bboxes[:, 0] /= kp_scale_x; bboxes[:, 2] /= kp_scale_x
                    bboxes[:, 1] /= kp_scale_y; bboxes[:, 3] /= kp_scale_y
                elif not is_seg_full_frame and last_keypoints is not None and len(last_keypoints) > 0:
                    # This det/pose full frame landed on a seg-skip frame --
                    # yolo_seg produced no fresh boxes here (seg_skip_n > 1).
                    # Fall back to keypoint-derived boxes (same pattern
                    # MobileSAM already uses for its own skip frames) instead
                    # of forcing a seg run seg_skip_n was meant to avoid.
                    kb = bboxes_from_keypoints(last_keypoints, last_scores, height, width, padding=40)
                    if kb:
                        kb_arr = np.array(kb, dtype=float)
                        bboxes = kb_arr.copy()
                        bboxes[:, 0] /= kp_scale_x; bboxes[:, 2] /= kp_scale_x
                        bboxes[:, 1] /= kp_scale_y; bboxes[:, 3] /= kp_scale_y
                    else:
                        bboxes = np.empty((0, 4), dtype=float)
                else:
                    # pose_model expects an ndarray (possibly empty), not None.
                    bboxes = np.empty((0, 4), dtype=float)
            else:
                t0     = time.time()
                # YOLO11n on the same 320x320 infer_frame the old YOLOX-Nano
                # det_model ran on -- boxes come back in that same 320x320
                # space, matching the kp_scale_x/kp_scale_y upscale below
                # unchanged.
                _, bboxes = person_detector.get_mask_and_boxes(infer_frame)
                if bboxes is None or len(bboxes) == 0:
                    bboxes = np.empty((0, 4), dtype=float)
                else:
                    bboxes = np.asarray(bboxes, dtype=float)
                t_det_total += time.time() - t0

            # Reject boxes too small relative to the frame's largest detection
            # to plausibly be the video's subject -- e.g. distant background
            # bystanders in a hallway shot that flicker in/out of detection as
            # they walk and otherwise get tracked/encrypted as short-lived
            # phantom person streams. An absolute pixel threshold doesn't work
            # here: the subject's own box size varies hugely with distance
            # from camera, so a bystander can be taller in pixels than the
            # subject is in another frame. Relative-to-largest adapts to that.
            if bboxes is not None and len(bboxes) > 1:
                box_area  = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
                max_area  = box_area.max()
                bboxes    = bboxes[box_area >= MIN_BOX_AREA_RATIO * max_area]

            last_bboxes = bboxes
            # scale detector bboxes from infer_frame space to original frame space
            if bboxes is not None and len(bboxes) > 0:
                sb = bboxes.copy().astype(float)
                sb[:, 0] *= kp_scale_x; sb[:, 2] *= kp_scale_x
                sb[:, 1] *= kp_scale_y; sb[:, 3] *= kp_scale_y
                last_scaled_bboxes = sb[:, :4].tolist()
            else:
                last_scaled_bboxes = []

            t2 = time.time()
            keypoints, scores = pose_model(infer_frame, bboxes=bboxes)
            t_pose_total += time.time() - t2

            if keypoints is not None and len(keypoints) > 0:
                keypoints[:, :, 0] *= kp_scale_x
                keypoints[:, :, 1] *= kp_scale_y

            last_keypoints  = keypoints
            last_scores     = scores
            last_mesh_cache = {}

            # Nearest-centroid identity matching -- see identity_tracker's
            # construction comment / PersonIdentityTracker's docstring for
            # why this replaced raw-index-based tracking. det_bboxes_this_frame
            # is reused below by the export block instead of recomputing the
            # same bboxes_from_keypoints() call a second time.
            id_detections = []
            det_bboxes_this_frame = {}
            if keypoints is not None and len(keypoints) > 0:
                for i in range(len(keypoints)):
                    db = bboxes_from_keypoints(
                        [keypoints[i]], [scores[i]], height, width, padding=40
                    )
                    if db:
                        x1, y1, x2, y2 = db[0]
                        id_detections.append((i, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
                        det_bboxes_this_frame[i] = [float(x1), float(y1), float(x2), float(y2)]

            identity_matches, departed = identity_tracker.assign(id_detections)
            # identity_matches: {identity_id: det_idx} -- who's present this
            # frame and which raw detection index they currently occupy.
            # Reverse map (det_idx -> identity_id) is what the export block
            # below needs, since it independently discovers a raw det_idx
            # via its own slot_tracker and must resolve it back to the
            # SAME identity_id person_states uses -- not a fresh index of
            # its own, which would silently be wrong again.
            det_idx_to_identity = {di: iid for iid, di in identity_matches.items()}
            last_det_idx_to_identity = det_idx_to_identity

            for dep_id in departed:
                if dep_id in person_states:
                    full_sid = person_states[dep_id].stream_id
                    enc_t, emb_t, gender_label, gender_conf = person_states[dep_id].flush_to_disk()
                    stream_gender_by_id[full_sid] = (gender_label, gender_conf)
                    t_encrypt_total += enc_t
                    t_embed_total   += emb_t
                    streams_flushed += 1
                    sid = full_sid[:8]
                    print(f"  [STREAM] Person {dep_id} departed -> "
                          f"stream {sid}... flushed "
                          f"(enc={enc_t*1000:.1f}ms emb={emb_t*1000:.1f}ms)")
                    del person_states[dep_id]

            if keypoints is not None and len(keypoints) > 0:
                for identity_id in identity_matches:
                    if identity_id not in person_states:
                        person_states[identity_id] = PersonState(
                            ttp_public_key,
                            export_crypto_dir if export_enabled else enc_output_dir,
                            embedder, benchmark=benchmark,
                            gender_classifier=gender_classifier,
                        )
                        print(f"  [STREAM] Person {identity_id} appeared -> "
                              f"stream {person_states[identity_id].stream_id[:8]}... created")

                per_person_movement = []

                for identity_id, i in identity_matches.items():
                    kpts  = keypoints[i]
                    scrs  = scores[i]
                    state = person_states[identity_id]

                    nose_xy = kpts[COCO_NOSE]
                    disp    = euclidean(nose_xy, state.prev_nose) if state.prev_nose else 0.0
                    state.prev_nose     = tuple(nose_xy)
                    state.movement_tier = get_movement_tier(disp, SLOW_THRESHOLD, FAST_THRESHOLD)
                    per_person_movement.append(disp)

                    if scrs[COCO_LEFT_EYE] > 0.3 and scrs[COCO_RIGHT_EYE] > 0.3:
                        state.inter_eye_px   = euclidean(kpts[COCO_LEFT_EYE], kpts[COCO_RIGHT_EYE])
                        state.face_size_tier = get_face_size_tier(
                            state.inter_eye_px, FAR_THRESHOLD, MEDIUM_THRESHOLD
                        )
                    else:
                        state.face_size_tier = "far"

                    t_fm0               = time.time()
                    face_crop_for_state = None
                    face_bbox_for_state = None
                    face_yaw_deg        = None  # None -> face_quality_from_yaw() not applied (see below)
                    face_eyes_nose_local = None  # crop-local (left_eye, right_eye, nose) -- see update_best()

                    if state.face_size_tier != "far":
                        crop, x_off, y_off, crop_dims, _ = derive_face_crop(frame, kpts, scrs)
                        if crop is not None:
                            face_crop_for_state = crop
                            face_bbox_for_state = (
                                x_off, y_off,
                                x_off + crop_dims[0], y_off + crop_dims[1],
                            )
                            # Re-expressed relative to the crop's own origin
                            # (not the full frame) since PersonState only
                            # retains best_face_crop, not the full frame it
                            # came from -- see tracking.py's update_best()/
                            # flush_to_disk() and gender.py's
                            # predict_from_keypoints() docstring for why
                            # eye-line alignment needs these at all.
                            face_eyes_nose_local = (
                                (kpts[COCO_LEFT_EYE][0] - x_off, kpts[COCO_LEFT_EYE][1] - y_off),
                                (kpts[COCO_RIGHT_EYE][0] - x_off, kpts[COCO_RIGHT_EYE][1] - y_off),
                                (kpts[COCO_NOSE][0] - x_off, kpts[COCO_NOSE][1] - y_off),
                            )
                            if face_canonicalizer is not None:
                                # selfie_seg mode: canonicalizer handles face detection.
                                # Canonical-face RENDERING stays person-0-only (expensive,
                                # and only person 0's expression is exported downstream --
                                # see the internal runbook), but yaw is cheap to also read for every
                                # tracked person here, since update_best()'s "best embedding
                                # source" ranking needs it for all of them, not just person 0.
                                # "Person 0" means identity_id == 0 -- the first identity
                                # PersonIdentityTracker ever assigned (stable across frames,
                                # unlike the old raw det_idx == 0 this used to check, which
                                # could silently refer to a different physical person from
                                # one frame to the next).
                                tc0 = time.time()
                                if identity_id == 0:
                                    cf, face_yaw_deg = face_canonicalizer.get_canonical_face_and_yaw(crop)
                                    if cf is not None:
                                        last_canonical_face = cf
                                else:
                                    _, face_yaw_deg = face_canonicalizer.get_canonical_face_and_yaw(crop)
                                t_canonical_total += time.time() - tc0
                            elif anonymizer == "convexhull":
                                # face_mesh_pts only feeds blur_all_persons()'s convex-hull
                                # region (see the blur dispatch below) -- yoloseg*/selfie_seg
                                # anonymizers blur from their own segmentation mask and never
                                # read face_mesh_pts, so running this MediaPipe FaceLandmarker
                                # call for them was pure wasted cost (confirmed: ~20ms/frame
                                # with zero effect on their output).
                                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                                result   = face_mesh.detect(mp_image)
                                if result.face_landmarks:
                                    pts = project_landmarks(
                                        result.face_landmarks[0],
                                        x_off, y_off, crop_dims[0], crop_dims[1],
                                    )
                                    state.face_mesh_pts = pts
                                    last_mesh_cache[i]  = pts
                                    if result.facial_transformation_matrixes:
                                        face_yaw_deg = yaw_from_transform(
                                            result.facial_transformation_matrixes[0]
                                        )
                    else:
                        state.face_mesh_pts = None
                    t_facemesh_total += time.time() - t_fm0

                    # last_scaled_bboxes[i] is this same person's detector box
                    # this frame (whichever detector actually ran -- YOLO11n
                    # (person_detector) or yolo_seg's own detect head, see the
                    # is_full_frame block above) -- used by derive_body_crop in
                    # place of its keypoint-only box, since no COCO-17 keypoint
                    # reaches the head/hair. Index-matched to keypoints/scores
                    # since both come from the same bboxes array passed into
                    # pose_model(). Guarded since last_scaled_bboxes can be
                    # shorter than keypoints in rare detector/pose-count
                    # mismatches -- falls back to the keypoint-only box via
                    # derive_body_crop's detector_bbox=None in that case.
                    detector_bbox = (
                        last_scaled_bboxes[i]
                        if last_scaled_bboxes is not None and i < len(last_scaled_bboxes)
                        else None
                    )
                    body_crop, bx1, by1, bx2, by2 = derive_body_crop(
                        frame, kpts, scrs, detector_bbox=detector_bbox
                    )
                    body_bbox  = (bx1, by1, bx2, by2) if body_crop is not None else None
                    confidence = compute_frame_confidence(scrs)
                    face_quality = (face_quality_from_yaw(face_yaw_deg)
                                    if face_yaw_deg is not None else 1.0)

                    state.update_best(
                        frame_idx    = frame_idx,
                        confidence   = confidence,
                        face_crop    = face_crop_for_state,
                        face_bbox    = face_bbox_for_state,
                        body_crop    = body_crop,
                        body_bbox    = body_bbox,
                        face_quality = face_quality,
                        face_eyes_nose = face_eyes_nose_local,
                    )
                    # Independent of update_best()'s single "best frame"
                    # selection -- restoring the real video on Tier 3
                    # approval needs EVERY frame's body crop, not just the
                    # one best-confidence frame used for the embedding. See
                    # tracking.py's append_frame() / flush_to_disk()'s
                    # extended .packet format.
                    state.append_frame(frame_idx, body_crop, body_bbox)

                if per_person_movement:
                    movement_tier = get_movement_tier(
                        max(per_person_movement), SLOW_THRESHOLD, FAST_THRESHOLD
                    )
                    if movement_adaptive:
                        current_N = SKIP_N[movement_tier]

            if export_enabled:
                # Stable left-to-right slots are a separate concept from
                # person_states above: export needs a small FIXED set of
                # identities (slot 0/1/2), while person_states is an
                # unbounded set that grows/shrinks with however many people
                # actually appear (see PersonIdentityTracker). Detections
                # beyond export_people, or too low-confidence to derive any
                # bbox, simply aren't exported -- they're still blurred
                # normally via the existing mask path.
                #
                # Reuses id_detections/det_bboxes_this_frame computed above
                # (same bboxes_from_keypoints() call over the same keypoints/
                # scores this frame) instead of recomputing an identical
                # detections list a second time.
                detections = id_detections
                det_bboxes = det_bboxes_this_frame

                slot_matches = slot_tracker.assign(detections)

                for s in range(export_people):
                    crop_s = None
                    if s in slot_matches:
                        di = slot_matches[s]
                        # Bridge: this slot's occupant this frame is whoever
                        # person_states tracks under the SAME stable identity
                        # this raw det_idx resolves to this frame (see
                        # det_idx_to_identity, built from the identity_tracker
                        # above -- NOT di itself, which is only this frame's
                        # raw detection index and was the actual source of a
                        # real bug: two different encrypted streams ended up
                        # both recording the same middle person's face in a
                        # 3-person clip, because di was being used directly
                        # as if it were a stable person_states key). Record
                        # its REAL crypto stream_id (see tracking.py's
                        # PersonState) so the manifest can point the phone
                        # at that stream's .packet/.key files. Overwritten
                        # each frame the slot is occupied; in practice a
                        # slot's occupant is stable for the clip (see
                        # ExportSlotTracker), so this converges to one real
                        # stream_id per slot, just derived live rather than
                        # invented at export start.
                        identity_id_for_slot = det_idx_to_identity.get(di)
                        if identity_id_for_slot in person_states:
                            export_slot_stream_id[s] = person_states[identity_id_for_slot].stream_id
                        kpts_s, scrs_s = keypoints[di], scores[di]
                        export_kp_rows[s].append(
                            np.concatenate(
                                [kpts_s[:, :2], scrs_s[:, None]], axis=1
                            ).astype(np.float32)
                        )
                        export_bbox_rows[s].append([det_bboxes[di]])
                        # The STABLE id of whoever occupies this slot on this
                        # frame -- the same one person_states is keyed by, not
                        # the raw det index. Buffered in lockstep with
                        # export_kp_rows so a whole-clip pose transform can
                        # split a re-let slot into per-person sequences instead
                        # of splicing two people onto one template and one seed.
                        export_identity_rows[s].append(identity_id_for_slot)

                        crop_s, _, _, _, _ = derive_face_crop(frame, kpts_s, scrs_s)
                        if crop_s is not None:
                            export_last_face_crop[s] = crop_s
                            params_s = export_face_canon.extract_params(crop_s)
                            if params_s is not None:
                                export_last_face_params[s] = params_s
                                export_valid_smiles[s].append(float(params_s[P_SMILE]))
                            # Gender is no longer classified per-frame here.
                            # It now comes from stream_gender_by_id (see the
                            # manifest writer below), populated once per
                            # stream at PersonState.flush_to_disk() time --
                            # the SAME single-best-frame classification the
                            # normal (non-export) path already used. Running
                            # it here too, every frame, was a second,
                            # independent per-frame-voting scheme that could
                            # disagree with the non-export path's answer for
                            # the exact same clip -- removed 2026-08-14 so
                            # both paths always produce the same result.
                    else:
                        export_kp_rows[s].append(np.zeros((17, 3), dtype=np.float32))
                        export_bbox_rows[s].append([])
                        # Empty slot: no identity. Appended on this path too so
                        # the two buffers can never drift out of lockstep.
                        export_identity_rows[s].append(None)

                    # face_params_p{i}.npy is the safe-to-transmit identity-free
                    # signal -- last-good scalars held across brief absences, same
                    # hold-over behaviour the old raw-crop export used, just now
                    # applied to 12 numbers instead of a face image.
                    if export_last_face_params[s] is not None:
                        export_face_param_rows[s].append(export_last_face_params[s].copy())
                    else:
                        export_face_param_rows[s].append(np.zeros(12, dtype=np.float32))

                    if export_diagnostics:
                        if export_last_face_crop[s] is not None:
                            face_frame = cv2.resize(
                                export_last_face_crop[s], (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE)
                            )
                        else:
                            face_frame = np.zeros(
                                (EXPORT_FACE_SIZE, EXPORT_FACE_SIZE, 3), dtype=np.uint8
                            )
                        export_face_writers[s].write(face_frame)

                if export_diagnostics:
                    overlay = frame.copy()
                    if last_scaled_bboxes:
                        for bbox in last_scaled_bboxes:
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    export_bbox_overlay_writer.write(overlay)

                if len(export_kp_rows[0]) % EXPORT_FLUSH_EVERY == 0:
                    _write_export_arrays()

            # Collect selfie-seg result (may already be done; blocks only if det+pose faster)
            if _future_seg is not None:
                raw_seg_mask       = _future_seg.result()
                seg_mask_keypoints = keypoints.copy() if keypoints is not None and len(keypoints) > 0 else None
                t_seg_total       += time.time() - _t_seg0  # wall-time waited, not thread time

                # Selfie-seg segments the whole frame with no notion of "where
                # RTMPose actually detected a person" -- it will happily paint
                # over background clutter (tree branches, textured walls) that
                # merely looks person-shaped. Gate the mask to the union of
                # detected person regions so nothing outside those regions can
                # ever be blurred; if no one was detected, apply no mask at all.
                #
                # last_scaled_bboxes comes from person_detector and can be []
                # even when keypoints is non-empty: rtmlib's RTMPose silently
                # falls back to a whole-frame box when given zero detector
                # boxes, so it
                # still produces a pose for a real, visible person the
                # detector merely missed on this frame. Gating on the empty
                # detector boxes alone would wipe the mask for a real person
                # still being tracked -- fall back to a keypoint-derived
                # region in that case instead of gating with nothing.
                #
                # But the fallback pose itself can also be low-confidence
                # (e.g. mid-stride motion blur): every keypoint scores below
                # kpt_thr, so bboxes_from_keypoints returns [] too. Both
                # signals failing on the same frame doesn't mean the person
                # vanished -- it means det+pose had a bad frame. Hold the
                # last known-good gate region for a short TTL instead of
                # collapsing to an all-False mask.
                if raw_seg_mask is not None:
                    if last_scaled_bboxes:
                        gate_bboxes = last_scaled_bboxes
                    elif keypoints is not None and len(keypoints) > 0:
                        gate_bboxes = bboxes_from_keypoints(
                            keypoints, scores, height, width, padding=40
                        )
                    else:
                        gate_bboxes = []

                    fresh_gate = bool(gate_bboxes)
                    if gate_bboxes:
                        region = bbox_region_mask(gate_bboxes, height, width, padding=40)
                        last_gate_region      = region
                        gate_region_stale_for = 0
                    elif last_gate_region is not None and gate_region_stale_for < GATE_REGION_TTL:
                        region = last_gate_region
                        gate_region_stale_for += 1
                    else:
                        region = bbox_region_mask([], height, width, padding=40)
                        last_gate_region       = None
                        gate_region_stale_for  = 0

                    last_seg_mask = raw_seg_mask & region

                    # Selfie-seg can also fail outright on a confidently-detected
                    # person: small/angled figures against a similarly-toned
                    # background sometimes make the confidence field come back
                    # near all-zero even inside a correct, fresh detector bbox.
                    # Gating can't help there since raw_seg_mask has nothing to
                    # gate. If that happens on a fresh (non-stale) detection,
                    # fall back to solid-filling the tight bbox itself -- a
                    # coarser silhouette beats leaving a real person unmasked.
                    if fresh_gate and last_seg_mask.sum() < 0.02 * region.sum():
                        last_seg_mask = bbox_region_mask(gate_bboxes, height, width, padding=0)

                    if export_enabled and export_diagnostics:
                        export_raw_mask_writer.write((raw_seg_mask.astype(np.uint8)) * 255)
                        export_gate_writer.write((region.astype(np.uint8)) * 255)
                else:
                    last_seg_mask = None
                    if export_enabled and export_diagnostics:
                        export_raw_mask_writer.write(np.zeros((height, width), dtype=np.uint8))
                        export_gate_writer.write(np.zeros((height, width), dtype=np.uint8))

            prev_gray = curr_gray.copy()

            if full_frame_count % TIMING_INTERVAL == 0:
                n = max(full_frame_count, 1)
                s = max(skip_frame_count, 1)
                f = max(frame_idx, 1)
                print(
                    f"[F{frame_idx:4d}] "
                    f"Det: {t_det_total/n*1000:5.1f}ms | "
                    f"Pose: {t_pose_total/n*1000:5.1f}ms | "
                    f"FaceMesh: {t_facemesh_total/n*1000:5.1f}ms | "
                    f"OF-body: {t_of_body_total/s*1000:4.1f}ms | "
                    f"OF-face: {t_of_face_total/s*1000:4.1f}ms | "
                    f"Blur: {t_blur_total/f*1000:4.1f}ms | "
                    f"N={current_N} People={len(keypoints) if keypoints is not None else 0}"
                )

        else:
            skip_frame_count += 1

            if last_keypoints is not None and len(last_keypoints) > 0:
                n_persons = len(last_keypoints)

                # --- body LK tasks (one per person) ---
                def _body_lk(i):
                    old = last_keypoints[i][:, :2].astype(np.float32).reshape(-1, 1, 2)
                    new, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old, None, **LK_PARAMS)
                    return i, new

                # Face LK only needed for convexhull blur (face_mesh_pts → hull)
                # When canonicalizer active, skip face LK and run canonicalizer instead
                face_mesh_inputs = {}
                if face_canonicalizer is None:
                    for i in range(n_persons):
                        identity_id = last_det_idx_to_identity.get(i)
                        if identity_id in person_states and person_states[identity_id].face_mesh_pts is not None:
                            face_mesh_inputs[i] = np.array(
                                person_states[identity_id].face_mesh_pts, dtype=np.float32
                            ).reshape(-1, 1, 2)

                def _face_lk(i, old_face):
                    new, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, old_face, None, **LK_PARAMS)
                    return i, new

                # Submit body LK and face LK in parallel
                t_lk0 = time.time()
                body_futures = [_lk_pool.submit(_body_lk, i) for i in range(n_persons)]
                face_futures = {i: _lk_pool.submit(_face_lk, i, old)
                                for i, old in face_mesh_inputs.items()}

                # Collect body results
                tracked_keypoints = [None] * n_persons
                for fut in body_futures:
                    i, new_pts = fut.result()
                    tk = last_keypoints[i].copy()
                    tk[:, :2] = new_pts.reshape(-1, 2)
                    tracked_keypoints[i] = tk
                t_of_body_total += time.time() - t_lk0

                # NOTE: tried independently LK-tracking the detector's own
                # box here (either single-center-point via tracking.py's
                # propagate_bboxes(), or 4-corner) so the anonymization box
                # would track the same way MIRAGE's does. Reverted: a
                # rigid box (however it's tracked) can only translate/scale,
                # it cannot grow to cover a limb extending outward between
                # full-detection frames the way a keypoint-derived box does
                # (bboxes_from_keypoints(), used elsewhere in this file) --
                # e.g. someone stretching an arm out would get clipped by a
                # box-tracked region but stay covered by a keypoint-derived
                # one. Independently-tracked box vs. keypoint-tracked pose
                # also visibly drifted apart from each other on weak-texture
                # backgrounds during testing (2026-08-14) -- two independent
                # LK tracks with nothing keeping them mutually consistent.
                # last_scaled_bboxes is intentionally left as the last real
                # detector output on skip frames (not re-tracked); callers
                # needing an up-to-date skip-frame box already fall back to
                # bboxes_from_keypoints() (see the is_full_frame block above
                # and the yolo_seg full-frame-landing case).

                # Collect face LK results (convexhull mode only)
                t_face0 = time.time()
                for i, fut in face_futures.items():
                    _, new_face = fut.result()
                    identity_id = last_det_idx_to_identity[i]
                    person_states[identity_id].face_mesh_pts = [
                        (int(pt[0][0]), int(pt[0][1])) for pt in new_face
                    ]
                t_of_face_total += time.time() - t_face0

                # Canonical runs only on full frames (hidden in selfie-seg parallel wait).
                # Skip frames reuse last_canonical_face - expression changes ~10fps is enough.

                keypoints      = np.array(tracked_keypoints)
                scores         = last_scores
                last_keypoints = keypoints

            prev_gray = curr_gray.copy()

        # emit_mask: THE mask this frame -- the one that is applied to the
        # frame and the one that is exported. It is `last_seg_mask` itself
        # unless the silhouette defence is on, in which case it is the
        # mitigated copy and `last_seg_mask` deliberately keeps the
        # UN-mitigated value.
        #
        # 🔴 That separation is the whole point. `last_seg_mask` is PROPAGATED
        # across frames (warped forward on skip frames and reused). Writing the
        # mitigated mask back into it would feed the mitigation's own temporal
        # running-max its own output, compounding frame over frame until the
        # grey region covers the frame. Keep them separate; never assign
        # emit_mask back into last_seg_mask.
        emit_mask = last_seg_mask
        tb0 = time.time()
        if blur_bodies:
            if mobile_sam is not None and keypoints is not None and len(keypoints) > 0:
                # Full frames: use detector bboxes (more accurate, includes head).
                # Skip frames: fall back to keypoint-derived bboxes.
                if is_full_frame and last_scaled_bboxes:
                    sam_bboxes = last_scaled_bboxes
                else:
                    sam_bboxes = bboxes_from_keypoints(
                        keypoints, scores, height, width, padding=80
                    )
                annotated = mobile_sam.blur_frame(annotated, sam_bboxes)
            elif selfie_seg is not None or yolo_seg is not None:
                # Full-frame mask already fetched in parallel above.
                # Skip frames: warp stored mask by affine from keypoint motion.
                if not is_full_frame and last_seg_mask is not None \
                        and seg_mask_keypoints is not None \
                        and keypoints is not None and len(keypoints) > 0:
                    old_pts = seg_mask_keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    new_pts = keypoints[:, :, :2].reshape(-1, 2).astype(np.float32)
                    M, _ = cv2.estimateAffinePartial2D(old_pts, new_pts, method=cv2.RANSAC)
                    if M is not None:
                        last_seg_mask = cv2.warpAffine(
                            last_seg_mask.astype(np.uint8), M, (width, height),
                            flags=cv2.INTER_NEAREST
                        ).astype(bool)
                    seg_mask_keypoints = keypoints.copy()
                emit_mask = last_seg_mask          # re-bound: the warp above may have replaced it
                if mask_anon is not None:
                    # MIRAGE silhouette mitigation (opt-in). Called on EVERY
                    # frame, drop frames included -- `apply(None)` is handled
                    # explicitly and keeps the temporal window ageing exactly as
                    # it does upstream, where a detection drop still appends an
                    # all-zero mask and still gets mitigated. Skipping the call
                    # on a drop frame would stall the ring buffer instead.
                    _tm0 = time.time()
                    emit_mask = mask_anon.apply(last_seg_mask)
                    t_mask_anon_total += time.time() - _tm0
                    # §2 accounting, measured here at the seam rather than taken
                    # on trust from the adapter: the emitted mask must be a
                    # SUPERSET of the pre-mitigation mask -- a pixel that was
                    # covered before must still be covered. Counted, never
                    # raised per frame; the count goes to the manifest so a run
                    # that violated it says so instead of failing mid-clip.
                    if last_seg_mask is not None and emit_mask is not None:
                        _viol = int(np.count_nonzero(last_seg_mask & ~emit_mask))
                        if _viol:
                            mask_superset_violations   += 1
                            mask_superset_violation_px += _viol
                if emit_mask is not None:
                    applier = selfie_seg if selfie_seg is not None else yolo_seg
                    annotated = applier.apply_mask(annotated, emit_mask)
            elif keypoints is not None and len(keypoints) > 0:
                all_face_mesh_pts = [
                    person_states[last_det_idx_to_identity[i]].face_mesh_pts
                    if i in last_det_idx_to_identity and last_det_idx_to_identity[i] in person_states
                    else None
                    for i in range(len(keypoints))
                ]
                annotated = blur_all_persons(annotated, keypoints, scores, all_face_mesh_pts)
        t_blur_total += time.time() - tb0

        if export_enabled:
            # Clean (debug-overlay-free) anonymized frame + final mask, for
            # a downstream machine consumer rather than a human viewer --
            # written here, before the skeleton/bbox debug drawing below.
            export_rtm_writer.write(annotated)
            if emit_mask is not None:
                export_mask_writer.write((emit_mask.astype(np.uint8)) * 255)
            else:
                export_mask_writer.write(np.zeros((height, width), dtype=np.uint8))
            if emit_mask_small_rows is not None:
                # A downscaled copy of exactly what was just applied+written,
                # for the per-slot stick-ink containment metric at flush time.
                # INTER_AREA + >= 0.5 means "a low-res cell counts as covered
                # when at least half of it was covered" -- a stated rule, not a
                # tuned one, and it neither grows nor shrinks the mask
                # systematically the way nearest-neighbour sampling can.
                if emit_mask is not None:
                    emit_mask_small_rows.append(
                        cv2.resize(emit_mask.astype(np.float32),
                                   (_mask_buf_w, _mask_buf_h),
                                   interpolation=cv2.INTER_AREA) >= 0.5)
                else:
                    emit_mask_small_rows.append(
                        np.zeros((_mask_buf_h, _mask_buf_w), dtype=bool))

        td0 = time.time()
        if draw_enabled and keypoints is not None and len(keypoints) > 0:
            annotated = draw_skeleton(annotated, keypoints, scores, kpt_thr=0.3)
            if not no_facemesh_draw:
                for i in range(len(keypoints)):
                    identity_id = last_det_idx_to_identity.get(i)
                    if identity_id in person_states and person_states[identity_id].face_mesh_pts:
                        draw_face_mesh_pts(annotated, person_states[identity_id].face_mesh_pts)
        if draw_enabled and last_scaled_bboxes:
            for bbox in last_scaled_bboxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        t_draw_total += time.time() - td0

        if not benchmark:
            now     = time.time()
            elapsed = max(now - prev_time, 1e-6)
            prev_time = now
            fps_history.append(1.0 / elapsed)
            if len(fps_history) > FPS_WINDOW:
                fps_history.pop(0)
            fps_display = sum(fps_history) / len(fps_history)

            if not no_draw and not no_hud:
                n_detected = len(keypoints) if keypoints is not None else 0
                frame_type = "SKIP(LK)" if not is_full_frame else "FULL"
                for j, line in enumerate([
                    f"FPS: {fps_display:.1f}",
                    f"Frame: {frame_type}",
                    f"People: {n_detected}",
                    f"Movement: {movement_tier}",
                    f"Skip-N: {current_N}",
                ]):
                    cv2.putText(annotated, line, (10, 35 + j * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

        tw0 = time.time()
        if out is not None:
            if face_canonicalizer is not None:
                # Three panels: original | blurred | canonical
                orig_panel    = cv2.resize(frame,    (DISPLAY_H, DISPLAY_H))
                blurred_panel = cv2.resize(annotated, (DISPLAY_H, DISPLAY_H))
                canon_panel   = np.full((DISPLAY_H, DISPLAY_H, 3), (228, 225, 222), dtype=np.uint8)
                if last_canonical_face is not None:
                    cf_resized = cv2.resize(last_canonical_face, (DISPLAY_H, DISPLAY_H))
                    canon_panel[:] = cf_resized
                _label = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(orig_panel,    "ORIGINAL",   (8, 24), _label, 0.6, (255, 255, 255), 2)
                cv2.putText(blurred_panel, "BLURRED",    (8, 24), _label, 0.6, (255, 255, 255), 2)
                cv2.putText(canon_panel,   "EXPRESSION", (8, 24), _label, 0.6, (40,  40,  40),  2)
                combined = np.hstack([orig_panel, blurred_panel, canon_panel])
                _write_q.put((out.write, combined))
            else:
                _write_q.put((out.write, annotated.copy()))
        t_write_total += time.time() - tw0

        if not headless:
            cv2.imshow("MIRAGE", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nUser quit.")
                break

        frame_idx += 1

    if fail_closed_mask is not None:
        t_seg_total += fail_closed_mask.t_detect

    print(f"\n  Flushing {len(person_states)} remaining active stream(s)...")
    for idx, state in person_states.items():
        enc_t, emb_t, gender_label, gender_conf = state.flush_to_disk()
        stream_gender_by_id[state.stream_id] = (gender_label, gender_conf)
        t_encrypt_total += enc_t
        t_embed_total   += emb_t
        streams_flushed += 1
        print(f"  [STREAM] Person {idx} (EOV) -> "
              f"stream {state.stream_id[:8]}... flushed "
              f"(enc={enc_t*1000:.1f}ms emb={emb_t*1000:.1f}ms)")
    person_states.clear()

    _seg_pool.shutdown(wait=False)
    _lk_pool.shutdown(wait=False)

    # Stop the read thread before releasing cap -- it may still be blocked
    # inside cap.read() (live camera) or _frame_q.put() (file, unbounded
    # queue can't block, but be defensive); signal it and drain one sentinel
    # slot so a live camera's already-in-flight read() call can return and
    # the thread can observe _read_stop and exit cleanly instead of calling
    # cap.read() on an about-to-be-released capture.
    _read_stop.set()
    _read_thread.join(timeout=2.0)

    cap.release()

    # Drain the write queue fully BEFORE releasing the VideoWriter -- unlike
    # the read queue (fine to drop stale work), every queued write must
    # actually land on disk, so this blocks until _write_worker has consumed
    # everything already queued, then signals it to exit via the sentinel.
    # Runs even when out is None (save_video=False, e.g. --benchmark/--no-save)
    # so the write thread is always shut down cleanly rather than left
    # relying on daemon=True to be reaped at process exit.
    _write_q.join()          # blocks until all queued frames are written
    _write_q.put(None)       # sentinel -- tells _write_worker to exit
    _write_thread.join(timeout=5.0)
    if out is not None:
        out.release()

    if export_enabled:
        _write_export_arrays()

        for i in range(export_people):
            if export_diagnostics:
                export_face_writers[i].release()

        export_rtm_writer.release()
        export_mask_writer.release()
        export_face_canon.close()
        if export_diagnostics:
            export_raw_mask_writer.release()
            export_gate_writer.release()
            export_bbox_overlay_writer.release()
        print(f"\n  Dense export written -> {export_dir}  "
              f"({frame_idx} frames x {export_people} slots)")
        if defence_cfg is not None:
            print(f"  Defence provenance   -> {os.path.join(export_dir, 'TIER1_CONFIG.json')}"
                  f"  (also in manifest.json as 'anon' / 'mask')")
            if mask_anon is not None:
                _ms = mask_anon.stats()
                print(f"  Silhouette           : {_ms['frames']} frames mitigated, "
                      f"area ratio mean={_ms['area_ratio_mean']}, "
                      f"§2 superset violations={mask_superset_violations} "
                      f"({mask_superset_violation_px} px)")
            if gait_anon:
                print("  Gait                 : see manifest.json -> anon.runtime "
                      "(raw_passthrough_frames is the number that matters)")
                # Surfaced here because it is silent everywhere else: the vendored
                # canvas clip absorbs a bone-length blow-up without raising, so a
                # non-zero count beside a small min_group_len_px means the emitted
                # skeleton was placed by np.clip, not by the transform. See
                # gait_anon._degenerate_groups for the measured table.
                _gp   = _last_gait_prov or {}
                _cl   = _gp.get("canvas_clipped_joints")
                _mgl  = _gp.get("min_group_len_px_over_sequences")
                # Printed UNCONDITIONALLY: this is the seed-independent number, and
                # a reader needs it even (especially) when the clip count is 0.
                print(f"  Gait min bone median : {_mgl} px (smallest per-group median over all "
                      f"transformed sequences; the vendored divisor is `median + 1e-6`, so a "
                      f"small value here means large limb amplification)")
                _clc = _gp.get("canvas_clipped_joints_confident")
                if _cl:
                    # Lead with the CONFIDENT count. The raw total also counts joints the
                    # detector never placed, and reading it alone turns a subject who is
                    # simply cut off by the camera into an apparent transform defect --
                    # measured on d07 (2 people, 50 f @1264²): raw 463 vs confident 42,
                    # every one of the 42 on the subject whose real box sat on the frame
                    # bottom in 50/50 frames.
                    _bad = [(s.get("slot"), s.get("min_group_len_px"),
                             s.get("canvas_clipped_joints_confident"),
                             "cut-off-by-framing" if s.get("subject_cut_off_by_framing")
                             else "fully-in-frame")
                            for s in _gp.get("sequences", [])
                            if s.get("canvas_clipped_joints")]
                    _sev = "!!" if _clc else "  "
                    print(f"  {_sev} Gait canvas clip  : {_clc} CONFIDENT emitted joints pinned to "
                          f"the frame edge from an input that was inside it ({_cl} counting "
                          f"unobserved joints too, which carry no information) -- placed by "
                          f"np.clip (pose_anon_edge.py:2426), not by the transform. "
                          f"(slot, min median bone px, confident clipped, framing): {_bad}. "
                          f"A sequence marked cut-off-by-framing has confident INPUT joints on "
                          f"the border already, so its clipping is the camera's, not the "
                          f"transform's. NOTE this count is seed-dependent and 0 does NOT mean "
                          f"clean; read the min bone median above.",
                          file=sys.stderr, flush=True)

    if not headless:
        cv2.destroyAllWindows()
    face_mesh.close()
    if selfie_seg is not None:
        selfie_seg.close()
    if face_canonicalizer is not None:
        face_canonicalizer.close()
    # MobileSAM has no explicit close; PyTorch model is GC'd

    total_time = time.time() - loop_start
    n  = max(full_frame_count, 1)
    s  = max(skip_frame_count, 1)
    f  = max(frame_idx, 1)
    sf = max(streams_flushed, 1)
    avg_fps = frame_idx / total_time

    print("\n" + "=" * 60)
    print("  FINAL TIMING SUMMARY")
    print("=" * 60)
    print(f"Benchmark mode      : {benchmark}")
    print(f"Movement adaptive   : {movement_adaptive}")
    print(f"Skip-N config       : {SKIP_N}")
    print(f"Total frames        : {frame_idx}")
    print(f"Full inf frames     : {full_frame_count}  ({full_frame_count/f*100:.1f}% of total)")
    print(f"Skip frames (LK)    : {skip_frame_count}  ({skip_frame_count/f*100:.1f}% of total)")
    print(f"Streams flushed     : {streams_flushed}")
    print(f"Total time          : {total_time:.1f}s")
    print(f"Average FPS         : {avg_fps:.2f}")
    print()
    print(f"-- Per-component (benchmark-clean) ------------------")
    # Labels below are anonymizer-aware: which code path actually produces
    # detection/mask/face-signal work differs per --anonymizer (see the
    # anonymizer-loading block above), so a fixed label set silently showed
    # 0.0ms for whichever path DIDN'T run under that exact name -- e.g.
    # yolo11n_boxfill's detection cost was previously invisible under
    # "Avg Det/full frame" (stayed 0.0ms) because it runs through yolo_seg's
    # own call, not person_detector's -- and showed up mislabeled under
    # "Avg SelfieSeg/full" instead, even though no MediaPipe SelfieSegmentation
    # was involved. Found + fixed 2026-08-14.
    if yolo_seg is not None:
        # person_detector's own call is skipped when yolo_seg is active (see
        # the is_full_frame block's duplicate-detection-avoidance comment) --
        # detection cost is entirely inside yolo_seg's own get_mask_and_boxes().
        print(f"Avg Detect+Mask/full: {t_seg_total      / n * 1000:.1f}ms  ({anonymizer}, includes detection -- person_detector not separately called)")
        if fail_closed_mask is not None:
            print(f"Avg EgoMotion/frame : {fail_closed_mask.t_ego / f * 1000:.1f}ms  (fail-closed mask, runs on every frame)")
    else:
        print(f"Avg Det/full frame  : {t_det_total      / n * 1000:.1f}ms  (person_detector, YOLO11n)")
        if selfie_seg is not None:
            print(f"Avg SelfieSeg/full  : {t_seg_total        / n * 1000:.1f}ms  (parallel w/ det+pose)")
        elif mobile_sam is not None:
            pass  # MobileSAM's own cost isn't separately timed into t_seg_total
    if face_canonicalizer is not None:
        print(f"Avg FaceSignal/full : {t_canonical_total / n * 1000:.1f}ms  (canonical expression -- full frames only, reused/held on skip)")
    elif anonymizer == "convexhull":
        print(f"Avg FaceSignal/full : {t_facemesh_total / n * 1000:.1f}ms  (raw FaceMesh for convex-hull blur region)")
    print(f"Avg OF-body/skip    : {t_of_body_total   / s * 1000:.1f}ms  (parallel w/ OF-face)")
    print(f"Avg OF-face/skip    : {t_of_face_total   / s * 1000:.1f}ms  (parallel w/ OF-body)")
    print(f"Avg Blur/frame      : {t_blur_total     / f * 1000:.1f}ms  (mask apply + warp only)")
    if mask_anon is not None:
        # Nested INSIDE the blur timer above, so this is a component of it, not
        # an addition to it.
        print(f"Avg MaskAnon/frame  : {t_mask_anon_total / f * 1000:.1f}ms  "
              f"(silhouette mitigation, included in Blur above)")
    if gait_anon and export_enabled:
        print(f"Total GaitAnon      : {t_gait_anon_total * 1000:.1f}ms  "
              f"(whole-buffer transform, re-run on every export flush)")
    if not benchmark:
        print(f"Avg Draw/frame      : {t_draw_total     / f * 1000:.1f}ms")
        print(f"Avg Write/frame     : {t_write_total    / f * 1000:.1f}ms")
        print(f"Avg Enc/stream      : {t_encrypt_total  / sf * 1000:.1f}ms")
        print(f"Avg Embed/stream    : {t_embed_total    / sf * 1000:.1f}ms")

    if csv_out is not None:
        file_exists = os.path.isfile(csv_out)
        with open(csv_out, 'a', newline='') as csvfile:
            fieldnames = [
                'input_file', 'anonymizer', 'benchmark', 'movement_adaptive', 'skip_n',
                'total_frames', 'full_frames', 'skip_frames',
                'full_pct', 'skip_pct', 'avg_fps',
                'det_ms', 'pose_ms', 'facemesh_ms',
                'of_body_ms', 'of_face_ms', 'blur_ms',
                'draw_ms', 'write_ms',
                'streams_flushed', 'total_time_s',
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'input_file'       : os.path.basename(input_path),
                'anonymizer'       : anonymizer,
                'benchmark'        : benchmark,
                'movement_adaptive': movement_adaptive,
                'skip_n'           : skip_n if not movement_adaptive else 'adaptive',
                'total_frames'     : frame_idx,
                'full_frames'      : full_frame_count,
                'skip_frames'      : skip_frame_count,
                'full_pct'         : round(full_frame_count / f * 100, 1),
                'skip_pct'         : round(skip_frame_count / f * 100, 1),
                'avg_fps'          : round(avg_fps, 2),
                'det_ms'           : round(t_det_total      / n * 1000, 1),
                'pose_ms'          : round(t_pose_total     / n * 1000, 1),
                'facemesh_ms'      : round(t_facemesh_total / n * 1000, 1),
                'of_body_ms'       : round(t_of_body_total  / s * 1000, 1),
                'of_face_ms'       : round(t_of_face_total  / s * 1000, 1),
                'blur_ms'          : round(t_blur_total      / f * 1000, 1),
                'draw_ms'          : round(t_draw_total      / f * 1000, 1),
                'write_ms'         : round(t_write_total     / f * 1000, 1),
                'streams_flushed'  : streams_flushed,
                'total_time_s'     : round(total_time, 2),
            })
        print(f"\nMetrics appended -> {csv_out}")

    if save_video and out is not None:
        print(f"\nOutput video: {output_path}")
