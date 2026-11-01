
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mirage.pipeline import process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRAGE RTMPose pipeline")

    parser.add_argument("input", type=str, help="Input video path")
    parser.add_argument("--output", type=str, default="/tmp/output_rtm.mp4",
                        help="Output video path (ignored in benchmark mode)")
    parser.add_argument("--enc-dir", type=str, default="data/output/encrypted",
                        help="Encrypted output directory")
    parser.add_argument("--skip-n", type=int, default=5,
                        help="Run full inference every N frames. Ignored if --movement-adaptive is set.")
    parser.add_argument("--movement-adaptive", action="store_true",
                        help="Enable movement-adaptive skip (slow/medium/fast tiers).")
    parser.add_argument("--benchmark", action="store_true",
                        help="Disable video save, crypto, and drawing. Measures pure pipeline latency.")
    parser.add_argument("--csv-out", type=str, default=None,
                        help="Append timing summary row to this CSV file.")
    parser.add_argument("--headless", action="store_true",
                        help="Disable display window (SSH/Pi mode).")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip VideoWriter.")
    parser.add_argument("--no-blur", action="store_true",
                        help="Disable anonymization.")
    parser.add_argument("--anonymizer", type=str, default="convexhull",
                        choices=["convexhull", "selfie_seg0", "selfie_seg1", "mobilesam", "yoloseg", "yoloseg11", "yoloseg11int8", "yoloseg11ncnn", "yolo11n_boxfill"],
                        help="Anonymization backend: convexhull (default), "
                             "selfie_seg0 (MediaPipe general), selfie_seg1 (MediaPipe landscape), "
                             "mobilesam (MobileSAM ViT-Tiny), yoloseg (YOLOv8n-seg ONNX), "
                             "yoloseg11 (YOLO11n-seg ONNX FP32), yoloseg11int8 (YOLO11n-seg ONNX INT8, dynamic quant), "
                             "yoloseg11ncnn (YOLO11n-seg NCNN FP32 -- ~1.96x faster than yoloseg11int8 on Pi 5), "
                             "yolo11n_boxfill (plain YOLO11n NCNN, detection-only, no segmentation -- rectangular "
                             "box grey-fill instead of a per-pixel mask; this is the detector actually measured in "
                             "the paper's Table 6/7 AP/AR results, see results/tier1_detection_eval/).")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="Dense per-person export mode: write per-frame keypoints/bboxes/"
                             "face-crops/masks into this directory (opt-in, additive).")
    parser.add_argument("--dense-export", action="store_true",
                        help="Force full inference on every frame (no skip/optical-flow). "
                             "Implied automatically when --export-dir is set.")
    parser.add_argument("--export-people", type=int, default=3,
                        help="Number of stable per-person export slots (default 3).")
    parser.add_argument("--export-diagnostics", action="store_true",
                        help="Also export raw_seg_mask.mp4, gate_region.mp4, bbox_overlay.mp4.")
    parser.add_argument("--seg-infer-size", type=int, default=320,
                        help="Network input size (px) for yoloseg* anonymizers. Lower = "
                             "faster (roughly quadratic), coarser mask boundaries. Ignored "
                             "for non-yoloseg anonymizers. Default 320 (unchanged behavior).")
    parser.add_argument("--segskip-n", type=int, default=1,
                        help="Run yoloseg* segmentation every N frames, independent of "
                             "--skip-n (which governs det/pose/face-canon). Skip frames "
                             "warp the last mask via affine-from-keypoint-motion (same "
                             "mechanism as --skip-n's own skip frames). 1 = segmentation "
                             "every frame (default, unchanged behavior). Ignored for "
                             "non-yoloseg anonymizers.")
    parser.add_argument("--no-draw", action="store_true",
                        help="Suppress the skeleton/facemesh debug overlay drawn on the "
                             "annotated/original panel (anonymization itself is unaffected).")
    parser.add_argument("--no-facemesh-draw", action="store_true",
                        help="Suppress only the 468-pt facemesh overlay; RTMPose body "
                             "skeleton keypoints are still drawn. Ignored if --no-draw is set.")
    parser.add_argument("--no-hud", action="store_true",
                        help="Suppress only the FPS/Frame/People/Movement/Skip-N corner "
                             "text; skeleton/facemesh overlays are unaffected. Ignored if "
                             "--no-draw is set.")
    parser.add_argument("--ttp-server", type=str, default=None,
                        help="Tier 3 TTP base URL, e.g. https://localhost:8843 -- required "
                             "unless --benchmark is set. Tier 1 fetches the TTP's public key "
                             "from this server rather than generating its own keypair.")
    parser.add_argument("--ttp-http", action="store_true",
                        help="Don't verify the Tier 3 server's TLS cert (matches its own "
                             "--http/self-signed-TOFU story for local testing).")

    # ---------------------------------------------------------------------
    # MIRAGE re-ID defences (vendored; src/mirage/vendor/mirage_edge/).
    # --gait-anon now defaults ON: the paper's Section 4
    # ("Tier 1: On-Device Privacy Enforcement") states pose anonymization
    # (canonical body proportions, track-consistent perturbation, trajectory
    # smoothing) as a mandatory, always-on step before egress, and Section 6
    # reports Table tab:reid_gait's numbers against it -- so the shipped
    # default now matches that. Use --no-gait-anon to reproduce the paper's
    # own "Quantized Pose Baseline" ablation row, which is explicitly
    # produced by disabling this transform.
    # NOTE: no privacy figure measured on the MIRAGE edge host describes this
    # pipeline. These flags name code and configuration, nothing more.
    # ---------------------------------------------------------------------
    defence = parser.add_argument_group("MIRAGE re-ID defences (gait-anon default ON)")
    defence.add_argument("--gait-anon", action="store_true", default=True,
                         help="Apply the vendored MIRAGE whole-clip gait/pose transform to the "
                              "EXPORTED keypoint arrays (keypoints_p*.npy). Requires "
                              "--export-dir: it does not alter the rendered video, the skeleton "
                              "overlay or the mask. On by default; use --no-gait-anon to disable.")
    defence.add_argument("--no-gait-anon", action="store_false", dest="gait_anon",
                         help="Disable the pose anonymization transform (see --gait-anon). "
                              "Needed to reproduce the paper's 'Quantized Pose Baseline' row, "
                              "which measures identity leakage with dynamic anonymization off.")
    defence.add_argument("--gait-preset", type=str, default="e2",
                         help="Vendored gait preset name (default: e2, the shipped arm). Pass an "
                              "empty string for the bare LEVELS behaviour with no preset kwargs. "
                              "An unknown name is refused, never silently defaulted.")
    defence.add_argument("--gait-on-degenerate", type=str, default="raise",
                         choices=["raise", "skip"],
                         help="What to do when a tracklet's median bone length for some joint "
                              "group is 0 px (e.g. the legs are unplaced for most of the "
                              "sequence): 'raise' stops rather than emit a skeleton nobody "
                              "validated (default); 'skip' leaves that sequence RAW on disk and "
                              "counts it in the manifest's raw_passthrough_frames. Neither is "
                              "free -- see gait_anon.py's DEFAULT_ON_DEGENERATE.")
    defence.add_argument("--gait-pin-run-seed", action="store_true",
                         help="TEST-ONLY. Pin ONE seed for the whole run instead of the vendored "
                              "PER-SEQUENCE draw. The default (unpinned) is the arm MIRAGE "
                              "measured and the only one that produces an export not stamped "
                              "'PRIVACY: TEST ARTIFACT - DO NOT SHIP'. Pinning consumes "
                              "MIRAGE_TEST_FIXED_SEED, which makes the perturbation shared by "
                              "every tracklet in the clip -- a linkable pseudo-identity (per-"
                              "identity seeding leaked 6-8x chance on every measured arm). It "
                              "was ALSO measured to buy nothing: 8.5 %% flush-to-flush spread "
                              "reduction, n=8, within noise. Owner decision 2026-08-14: default "
                              "flipped to unpinned. See pipeline.py's gait_pin_run_seed block.")
    defence.add_argument("--mask-shape-mode", type=str, default="none",
                         help="Vendored silhouette shape mode applied at the mask seam. 'none' "
                              "(default) = defence off. 'bbox' is the measured arm (a "
                              "segmentation backend + MIRAGE bbox). Requires a segmentation "
                              "backend: selfie_seg*, yoloseg*, or yolo11n_boxfill -- convexhull "
                              "and mobilesam are refused at startup, since neither keeps a "
                              "persistent mask to mitigate.")
    defence.add_argument("--mask-temporal-win", type=int, default=2,
                         help="Running-max window IN FRAMES for the silhouette mitigation. "
                              "DEFAULT 2 -- pinned (owner decision 2026-08-14), because it is the "
                              "window A.6o measured 'bbox' at, and because it is what covers a "
                              "dropped detection: measured on p01_c02.mp4 at win=1 the emitted "
                              "mask is EMPTY on the one drop frame (person unmasked); win=2 "
                              "covers it for +0.98 %% median emitted area. Pass 0 to derive it "
                              "from fps instead as max(1, round(0.14 s x emitted-mask fps)) -- "
                              "4 at 30 fps, 2 at 15, and 1 at 10, where 1 is the rung measured to "
                              "give the least protection and prints a loud warning.")
    defence.add_argument("--score-binarize-thresh", type=float, default=0.5,
                         help="Threshold at which the EXPORTED confidence column is collapsed to "
                              "{0,1} (default 0.5 == MIRAGE POSE_THRESH). Binarization itself "
                              "follows --gait-anon, because the raw per-joint confidence trace "
                              "is an identity side-channel that the vendored emit binarizes "
                              "alongside the pose. Affects the exported arrays ONLY -- the "
                              "in-loop consumers keep their own 0.3 threshold. "
                              "!! IT IS ALSO THE GAIT DEFENCE'S PRESENCE GATE: the same number "
                              "is passed to the adapter as conf_thresh, and a row whose every "
                              "joint falls below it is left UNTRANSFORMED (raw gait on disk, "
                              "counted in the manifest's raw_passthrough_breakdown."
                              "low_confidence_rows). The vendored unobserved-joint gate is "
                              "hardcoded at 0.5, so leave this at 0.5 unless you mean to move "
                              "both.")

    defence.add_argument("--fail-closed", action="store_true",
                         help="Fail-closed box mask (src/mirage/fail_closed.py): a low mask "
                              "threshold, the next detection warped back over each gap, and "
                              "unmatched boxes held for 2 more detector cycles. Boxes follow a "
                              "global ego-motion affine on every frame and the mask runs one "
                              "detector gap behind the camera. --segskip-n sets that gap: the "
                              "measured arm is 6. yolo11n_boxfill only, refused otherwise. Pose "
                              "keeps conf 0.40; mask.mp4 carries the extra boxes.")
    defence.add_argument("--fail-closed-conf", type=float, default=0.05,
                         help="Mask threshold under --fail-closed (default 0.05).")

    args = parser.parse_args()

    process_video(
        input_path        = args.input,
        output_path       = args.output,
        blur_bodies       = not args.no_blur,
        enc_output_dir    = args.enc_dir,
        headless          = args.headless,
        save_video        = not args.no_save,
        benchmark         = args.benchmark,
        skip_n            = args.skip_n,
        movement_adaptive = args.movement_adaptive,
        csv_out           = args.csv_out,
        anonymizer        = args.anonymizer,
        export_dir         = args.export_dir,
        dense_export        = args.dense_export,
        export_people        = args.export_people,
        export_diagnostics  = args.export_diagnostics,
        seg_infer_size      = args.seg_infer_size,
        seg_skip_n          = args.segskip_n,
        no_draw             = args.no_draw,
        no_facemesh_draw    = args.no_facemesh_draw,
        no_hud              = args.no_hud,
        ttp_server          = args.ttp_server,
        ttp_verify_tls      = not args.ttp_http,
        gait_anon             = args.gait_anon,
        gait_preset           = args.gait_preset,
        gait_on_degenerate    = args.gait_on_degenerate,
        gait_pin_run_seed     = args.gait_pin_run_seed,
        mask_shape_mode       = args.mask_shape_mode,
        mask_temporal_win     = args.mask_temporal_win,
        score_binarize_thresh = args.score_binarize_thresh,
        fail_closed           = args.fail_closed,
        fail_closed_conf      = args.fail_closed_conf,
    )
