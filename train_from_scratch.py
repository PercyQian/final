import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 from scratch (no pretrained weights) for CS:GO/CS2 roles."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configuration_files/custom_data.yaml",
        help="Path to data YAML (train/val paths, nc, names).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.yaml",
        help="YOLOv8 model yaml to initialize from scratch (e.g. yolov8n.yaml, yolov8s.yaml).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device id (e.g. '0') or 'cpu'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="model-results",
        help="Project directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="csgo_char_scratch",
        help="Run name under project directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (epochs without improvement).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    Path(args.project).mkdir(parents=True, exist_ok=True)

    # Initialize a YOLOv8 model FROM SCRATCH (no pretrained backbone)
    model = YOLO(args.model)

    # Train
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=False,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
        verbose=True,
    )

    # Export best weights path
    run_dir = Path(args.project) / args.name
    best_pt = run_dir / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best_pt if best_pt.exists() else 'not found'}")


if __name__ == "__main__":
    main()


