import cv2 as cv
import argparse
from ultralytics import YOLO


def drawPred(frame, bboxes, labels, confidences):
    for i, box in enumerate(bboxes):
        label = f"{labels[i]}: {confidences[i]:.2f}"

        x, y, w, h = map(int, box)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(y, labelSize[1])

        cv.rectangle(frame, (x, top - labelSize[1]),
                     (x + labelSize[0], top + baseLine),
                     (255, 255, 255), cv.FILLED)

        cv.putText(frame, label, (x, top),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),1)


def detect_objects(frame, model, threshold):
    results = model(frame, imgsz=640)[0]

    bboxes, labels, confidences = [], [], []

    for box in results.boxes:
        conf = float(box.conf[0])

        if conf > threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            label = model.names[cls]

            bboxes.append((x1, y1, x2 - x1, y2 - y1))
            labels.append(label)
            confidences.append(conf)

    return bboxes, labels, confidences


def process(args):

    model = YOLO("best.pt")

    stream = cv.VideoCapture(args.video)

    if not stream.isOpened():
        print("Error: Could not open video.")
        return

    fps_input = stream.get(cv.CAP_PROP_FPS)
    total_frames = int(stream.get(cv.CAP_PROP_FRAME_COUNT))

    delay = int(1000 / fps_input) if fps_input > 0 else 30

    frame_width = int(stream.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(stream.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(args.output, fourcc, fps_input,
                         (frame_width, frame_height))

    window_name = "YOLO Tracking"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, 900, 700)

    multi_tracker = cv.legacy.MultiTracker_create()

    DETECT_EVERY_N_FRAMES = 10
    SKIP_FRAMES = 2   # 🔥 automatic frame skipping

    frame_count = 0
    bboxes, labels, confidences = [], [], []
    ok = False

    paused = False
    last_frame = None
    current_frame_idx = 0

    while stream.isOpened():

        # 🔥 Skip frames automatically (only when not paused)
        if not paused and frame_count % SKIP_FRAMES != 0:
            stream.grab()
            frame_count += 1
            continue

        if not paused:
            grabbed, frame = stream.read()
            if not grabbed:
                break
            current_frame_idx += 1
        else:
            frame = last_frame.copy()

        timer = cv.getTickCount()

        if not paused:
            if frame_count % DETECT_EVERY_N_FRAMES == 0 or not ok:

                bboxes, labels, confidences = detect_objects(frame, model, args.thr)

                multi_tracker = cv.legacy.MultiTracker_create()

                for box in bboxes:
                    tracker = cv.legacy.TrackerKCF_create()
                    multi_tracker.add(tracker, frame, box)

                print("Detecting:", labels)

            else:
                if len(bboxes) > 0:
                    ok, bboxes = multi_tracker.update(frame)
                else:
                    ok = False

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        if bboxes:
            drawPred(frame, bboxes, labels, confidences)

            cv.putText(frame, f"FPS: {int(fps)}", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv.putText(frame, "No objects", (20, 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        last_frame = frame.copy()

        if paused:
            h, w = frame.shape[:2]
            cv.putText(frame, "PAUSED", (w - 180, h - 20),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        cv.imshow(window_name, frame)

        key = cv.waitKey(delay if not paused else 0) & 0xFF

        if key == 27:
            break
        elif key == 32:
            paused = not paused

        # ⏪ rewind still works
        elif key == ord('r'):
            rewind_frames = int(fps_input * 10)

            current_pos = int(stream.get(cv.CAP_PROP_POS_FRAMES))

            new_pos = max(0, current_pos - rewind_frames)
            stream.set(cv.CAP_PROP_POS_FRAMES, new_pos)


        frame_count += 1

    stream.release()
    out.release()
    cv.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(description='YOLO Tracking (Optimized)')

    parser.add_argument('--thr', type=float, default=0.4)
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.mp4')

    args = parser.parse_args()

    process(args)


if __name__ == '__main__':
    main()