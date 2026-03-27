import cv2 as cv
import numpy as np
import os
import math
import time
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# load sequence (frames + ground truth)


def load_sequence(seq_path):
    img_dir = os.path.join(seq_path, "img")
    gt_path = os.path.join(seq_path, "groundtruth_rect.txt")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")

    # Load frames
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(exts)
    ])

    if not frames:
        raise ValueError(f"No image files found in: {img_dir}")

    # Load ground truth
    gt_boxes = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Normalise separators then split
            vals = line.replace(",", " ").replace("\t", " ").split()
            gt_boxes.append([float(v) for v in vals[:4]])

    gt_boxes = np.array(gt_boxes)  # shape (N, 4)

    # Align lengths and take the shorter of the two
    n = min(len(frames), len(gt_boxes))
    return frames[:n], gt_boxes[:n]


def compute_iou(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter

    return inter / (union + 1e-6)


def compute_centre_error(box_pred, box_gt):
    cx_p = box_pred[0] + box_pred[2] / 2
    cy_p = box_pred[1] + box_pred[3] / 2
    cx_g = box_gt[0]   + box_gt[2]   / 2
    cy_g = box_gt[1]   + box_gt[3]   / 2
    return math.sqrt((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2)


def box_centre(box):
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def create_kalman():
    kf = cv.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    # Lower process noise means trusts its own motion model more
    kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
    # Higher measurement noise means trusts detector less
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    return kf


def run_kalman(frames, gt_boxes, show=False):
    print("\n  Running Kalman Filter ")

    kf = create_kalman()

    # Initialise state at GT centre of frame 1
    cx0, cy0 = box_centre(gt_boxes[0])
    kf.statePre = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)

    iou_scores    = []
    centre_errors = []
    fps_list      = []

    for i in range(1, len(frames)):
        frame  = cv.imread(frames[i])
        gt     = gt_boxes[i]

        t0 = time.time()

        # Predict
        pred_state = kf.predict()
        pred_cx    = float(pred_state[0][0])
        pred_cy    = float(pred_state[1][0])

        # Correct with simulated detector
        meas_cx, meas_cy = box_centre(gt)
        meas = np.array([[meas_cx], [meas_cy]], dtype=np.float32)
        kf.correct(meas)

        fps_list.append(1.0 / (time.time() - t0 + 1e-9))

        #Build predicted box using GT size
        pred_box = (
            pred_cx - gt[2] / 2,
            pred_cy - gt[3] / 2,
            gt[2],
            gt[3]
        )

        iou_scores.append(compute_iou(pred_box, gt))
        centre_errors.append(compute_centre_error(pred_box, gt))

        if show:
            _draw_frame(frame, pred_box, gt, iou_scores[-1], "Kalman", (0, 255, 255))
            cv.imshow("Kalman Tracker", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

    if show:
        cv.destroyAllWindows()

    print(f"Done. Avg FPS: {np.mean(fps_list):.1f}")
    return iou_scores, centre_errors, float(np.mean(fps_list))


def run_visual_tracker(tracker_name, frames, gt_boxes, show=False):
    print(f"\n  [{'2' if tracker_name == 'KCF' else '3'}/3] Running {tracker_name}...")

    if tracker_name == "KCF":
        tracker = cv.TrackerKCF_create()
    elif tracker_name == "CSRT":
        tracker = cv.legacy.TrackerCSRT_create()
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    first_frame = cv.imread(frames[0])
    init_box    = tuple(map(int, gt_boxes[0]))
    tracker.init(first_frame, init_box)

    iou_scores    = []
    centre_errors = []
    fps_list      = []

    for i in range(1, len(frames)):
        frame = cv.imread(frames[i])
        gt    = gt_boxes[i]

        t0 = time.time()
        success, pred_box = tracker.update(frame)
        fps_list.append(1.0 / (time.time() - t0 + 1e-9))

        if success:
            iou_scores.append(compute_iou(pred_box, gt))
            centre_errors.append(compute_centre_error(pred_box, gt))
        else:
            iou_scores.append(0.0)
            centre_errors.append(float("inf"))

        if show and success:
            _draw_frame(frame, pred_box, gt, iou_scores[-1], tracker_name,
                        (0, 0, 255) if tracker_name == "KCF" else (255, 100, 0))
            cv.imshow(f"{tracker_name} Tracker", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

    if show:
        cv.destroyAllWindows()

    print(f"     Done. Avg FPS: {np.mean(fps_list):.1f}")
    return iou_scores, centre_errors, float(np.mean(fps_list))


def _draw_frame(frame, pred_box, gt_box, iou, label, colour):
    # Ground truth: green
    xg, yg, wg, hg = map(int, gt_box)
    cv.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)
    cv.putText(frame, "GT", (xg, yg - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Prediction: tracker colour
    xp, yp, wp, hp = map(int, pred_box)
    cv.rectangle(frame, (xp, yp), (xp + wp, yp + hp), colour, 2)
    cv.putText(frame, label, (xp, yp - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    cv.putText(frame, f"IoU: {iou:.2f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def success_curve(iou_scores):
    thresholds = np.linspace(0, 1, 101)
    scores     = np.array(iou_scores)
    rates      = [float(np.mean(scores >= t)) for t in thresholds]
    auc        = float(np.mean(rates))
    return thresholds, rates, auc


def precision_curve(centre_errors):
    thresholds = np.linspace(0, 50, 101)
    errors     = np.array([e if e != float("inf") else 1e9 for e in centre_errors])
    rates      = [float(np.mean(errors <= t)) for t in thresholds]
    idx_20     = int(np.argmin(np.abs(thresholds - 20)))
    prec20     = rates[idx_20]
    return thresholds, rates, prec20


def plot_results(results, seq_name, save_path="tracker_comparison.png"):
    COLOURS = {
        "Kalman": "#f39c12",   # amber
        "KCF":    "#e74c3c",   # red
        "CSRT":   "#2980b9",   # blue
    }
    STYLES = {
        "Kalman": "--",
        "KCF":    "-",
        "CSRT":   "-",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"OTB Evaluation : Sequence: {seq_name}\n"
        f"(Kalman uses simulated perfect detector; KCF & CSRT are fully visual)",
        fontsize=11, fontweight="bold"
    )

    # Left: Success Plot
    for name, data in results.items():
        label = f"{name}  [AUC = {data['auc']:.3f}]"
        ax1.plot(data["suc_thresh"], data["suc_rates"],
                 label=label,
                 color=COLOURS.get(name, "gray"),
                 linestyle=STYLES.get(name, "-"),
                 linewidth=2.2)

    ax1.set_title("Success Plot", fontsize=11)
    ax1.set_xlabel("Overlap Threshold (IoU)")
    ax1.set_ylabel("Success Rate")
    ax1.set_xlim([0, 1]); ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Right: Precision Plot 
    for name, data in results.items():
        label = f"{name}  [Prec@20px = {data['prec20']:.3f}]"
        ax2.plot(data["pre_thresh"], data["pre_rates"],
                 label=label,
                 color=COLOURS.get(name, "gray"),
                 linestyle=STYLES.get(name, "-"),
                 linewidth=2.2)

    ax2.axvline(x=20, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)
    ax2.text(21, 0.05, "20px", color="gray", fontsize=8)
    ax2.set_title("Precision Plot", fontsize=11)
    ax2.set_xlabel("Location Error Threshold (pixels)")
    ax2.set_ylabel("Precision")
    ax2.set_xlim([0, 50]); ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # Kalman note patch 
    note = mpatches.Patch(color="#f39c12", alpha=0.4,
                          label="Kalman dashed = motion model only (not a fair comparison)")
    fig.legend(handles=[note], loc="lower center", fontsize=8, framealpha=0.5)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {save_path}")
    plt.show()


def print_summary(results, seq_name):
    print(f"  UNIFIED EVALUATION RESULTS : Sequence: {seq_name}")
    print(f"  {'Tracker':<12} {'AUC (Success)':>14} {'Precision@20px':>16} {'Avg FPS':>10}")


    for name, data in results.items():
        note = "  (simulated detector)" if name == "Kalman" else ""
        print(f"  {name:<12} {data['auc']:>14.4f} {data['prec20']:>16.4f} "
              f"{data['avg_fps']:>10.1f}{note}")


    # Best among visual trackers 
    visual = {k: v for k, v in results.items() if k != "Kalman"}
    if visual:
        best_auc  = max(visual, key=lambda k: visual[k]["auc"])
        best_prec = max(visual, key=lambda k: visual[k]["prec20"])
        best_fps  = max(visual, key=lambda k: visual[k]["avg_fps"])
        print(f"\n  Best AUC (visual trackers): {best_auc}")
        print(f"  Best Precision@20px (visual): {best_prec}")
        print(f"  Fastest visual tracker: {best_fps}")


def save_csv(results, seq_name, save_path="tracker_results.csv"):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "Tracker", "AUC (Success)",
                         "Precision@20px", "Avg FPS", "Notes"])
        for name, data in results.items():
            note = "Simulated perfect detector : motion model only" \
                   if name == "Kalman" else "Visual tracker : appearance based"
            writer.writerow([
                seq_name, name,
                round(data["auc"],    4),
                round(data["prec20"], 4),
                round(data["avg_fps"], 1),
                note
            ])
    print(f"  CSV saved:{save_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Unified OTB Evaluator : Kalman vs KCF vs CSRT"
    )
    parser.add_argument(
        "--seq", type=str, required=True,
        help="Path to OTB sequence folder (must contain img/ and groundtruth_rect.txt)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show live tracking window while evaluating (ESC to skip a tracker)"
    )
    parser.add_argument(
        "--output_plot", type=str, default="tracker_comparison.png",
        help="Output filename for the comparison plot (default: tracker_comparison.png)"
    )
    parser.add_argument(
        "--output_csv", type=str, default="tracker_results.csv",
        help="Output filename for the results CSV (default: tracker_results.csv)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.seq):
        print(f"Error: Sequence path not found:{args.seq}")
        return

    seq_name = os.path.basename(args.seq.rstrip("/\\"))

    print(f"  Unified OTB Tracker Evaluator")
    print(f"  Sequence : {seq_name}")
    print(f"  Path     : {args.seq}")

    # Load data 
    frames, gt_boxes = load_sequence(args.seq)
    print(f"  Frames loaded : {len(frames)}")

    # Run all three trackers
    results = {}

    iou_k, err_k, fps_k = run_kalman(frames, gt_boxes, show=args.show)
    iou_kcf, err_kcf, fps_kcf = run_visual_tracker("KCF",  frames, gt_boxes, show=args.show)
    iou_csr, err_csr, fps_csr = run_visual_tracker("CSRT", frames, gt_boxes, show=args.show)

    for name, ious, errs, fps in [
        ("Kalman", iou_k,   err_k,   fps_k),
        ("KCF",    iou_kcf, err_kcf, fps_kcf),
        ("CSRT",   iou_csr, err_csr, fps_csr),
    ]:
        st, sr, auc   = success_curve(ious)
        pt, pr, prec  = precision_curve(errs)
        results[name] = {
            "suc_thresh": st, "suc_rates": sr, "auc":    auc,
            "pre_thresh": pt, "pre_rates": pr, "prec20": prec,
            "avg_fps":    fps
        }

    # Output 
    print_summary(results, seq_name)
    plot_results(results, seq_name, save_path=args.output_plot)
    save_csv(results, seq_name, save_path=args.output_csv)


if __name__ == "__main__":
    main()