import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect CS:GO/CS2 roles in a video and export frame-level CSV and video-level summary."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8_csgo_cs2_model.pt",
        help="Path to YOLOv8 weights (.pt). Use trained weights in model-results/.../weights/best.pt for your run.",
    )
    parser.add_argument(
        "--source", type=str, required=True, help="Path to input video file (e.g., test.mp4)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs/video_infer",
        help="Output directory root",
    )
    parser.add_argument(
        "--save-vid",
        action="store_true",
        help="Save annotated output video",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_class_names(model: YOLO) -> Dict[int, str]:
    # Ultralytics stores names in model.names
    names = model.model.names if hasattr(model, "model") else model.names
    # Ensure int keys
    return {int(k): v for k, v in names.items()}


def draw_box(
    frame: np.ndarray, xyxy: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]
) -> None:
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 6, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    # Output paths
    time_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.out_dir) / f"{source_path.stem}_{time_tag}"
    ensure_dir(out_root)

    csv_path = out_root / "detections.csv"
    summary_path = out_root / "summary.json"
    out_video_path = out_root / f"{source_path.stem}_annotated.mp4"

    # Load model
    model = YOLO(args.weights)
    class_names = get_class_names(model)

    # If your dataset has a dummy class like 'none', we can optionally ignore it
    ignore_classes = {"none"}

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else -1

    writer = None
    if args.save_vid:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    rows: List[Dict] = []
    class_counts: Dict[str, int] = defaultdict(int)
    unique_classes: set = set()

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Run inference on the frame
        results = model.predict(source=frame, conf=args.conf, verbose=False)
        if not results:
            frame_index += 1
            continue

        res = results[0]
        if res.boxes is None:
            frame_index += 1
            if writer is not None:
                writer.write(frame)
            continue

        # Collect detections
        for i in range(len(res.boxes)):
            cls_idx = int(res.boxes.cls[i])
            conf = float(res.boxes.conf[i])
            x1, y1, x2, y2 = [int(v) for v in res.boxes.xyxy[i].tolist()]

            cls_name = class_names.get(cls_idx, str(cls_idx))
            if cls_name in ignore_classes:
                continue

            # Record row
            rows.append(
                {
                    "frame": frame_index,
                    "time_s": round(frame_index / fps, 3),
                    "class": cls_name,
                    "conf": round(conf, 4),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            class_counts[cls_name] += 1
            unique_classes.add(cls_name)

            # Draw on frame if saving video
            if writer is not None:
                color = (0, 0, 255) if "head" in cls_name else (0, 255, 0)
                draw_box(frame, (x1, y1, x2, y2), f"{cls_name} {conf:.2f}", color)

        if writer is not None:
            writer.write(frame)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    # Save CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    # Save summary
    duration_s = round((frame_index / fps) if fps > 0 else 0.0, 3)
    summary = {
        "video": str(source_path),
        "weights": str(args.weights),
        "total_frames": frame_index if total_frames < 0 else total_frames,
        "fps": fps,
        "duration_s": duration_s,
        "unique_classes": sorted(list(unique_classes)),
        "class_counts": dict(sorted(class_counts.items(), key=lambda x: x[0])),
        "csv_path": str(csv_path) if rows else None,
        "annotated_video": str(out_video_path) if args.save_vid else None,
        "conf_threshold": args.conf,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Finished. Summary saved to: {summary_path}")
    if rows:
        print(f"Detections CSV saved to: {csv_path}")
    if args.save_vid:
        print(f"Annotated video saved to: {out_video_path}")


if __name__ == "__main__":
    main()


