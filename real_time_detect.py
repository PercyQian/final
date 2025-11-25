import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

USE_DXCAM = True
import dxcam

def main():
    weights = "model-results/csgo_char_scratch/weights/best.pt"  # 改成你的权重
    conf_thres = 0.25
    device = "0"  # GPU 0；无GPU可用 "cpu"
    # 采集区域：None 全屏；或指定矩形 {"top":y,"left":x,"width":w,"height":h}
    region = None

    model = YOLO(weights)
    # 提前 warmup
    _ = model.predict(np.zeros((640, 640, 3), dtype=np.uint8), conf=conf_thres, device=device, verbose=False)

    save_dir = Path("runs/screen_infer")
    save_dir.mkdir(parents=True, exist_ok=True)


    cam = dxcam.create(device_idx=0, output_color="BGR")
    if region:
        cam.start(region=(region["left"], region["top"], region["left"]+region["width"], region["top"]+region["height"]))
    else:
        cam.start()
    grab = lambda: cam.get_latest_frame()

    fps_hist = []
    frame_id = 0
    print("Press 'q' to quit, 's' to save current prediction as image+YOLO label.")
    while True:
        t0 = time.time()
        frame = grab()
        if frame is None:
            continue

        # YOLO 推理（numpy BGR）
        results = model.predict(source=frame, conf=conf_thres, device=device, verbose=False)
        res = results[0]
        annotated = res.plot()  # 已叠加框/标签的 BGR

        # 显示
        if fps_hist:
            fps_txt = f"FPS: {1/np.mean(fps_hist):.1f}"
            cv2.putText(annotated, fps_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Screen Realtime Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            # 保存截图与预测为 YOLO 标签（弱标注）
            img_path = save_dir / f"frame_{frame_id:06d}.jpg"
            lbl_path = save_dir / f"frame_{frame_id:06d}.txt"
            cv2.imwrite(str(img_path), frame)
            # 将预测框保存为 YOLO 格式 (cls cx cy w h，相对坐标)
            if res.boxes is not None and len(res.boxes) > 0:
                h, w = frame.shape[:2]
                lines = []
                for i in range(len(res.boxes)):
                    cls_idx = int(res.boxes.cls[i])
                    x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"Saved: {img_path}, {lbl_path}")
            else:
                print("No boxes to save.")

        frame_id += 1
        fps = time.time() - t0
        fps_hist.append(fps)
        if len(fps_hist) > 30:
            fps_hist.pop(0)

    if USE_DXCAM:
        cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()