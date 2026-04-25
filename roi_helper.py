"""
ROI 校正工具
────────────
執行方式：
    python roi_helper.py                    # 使用 config.py 的串流 URL
    python roi_helper.py traffic_test.mp4  # 使用本機影片

操作說明：
    - 視窗出現後，先點 4 個點定義「左車道」ROI，再點 4 個點定義「右車道」ROI
    - 完成後按 Enter，座標會印在 terminal，複製貼到 config.py 即可
    - 按 R 重設，按 Q 離開
"""

import sys
import cv2
import numpy as np
from config import STREAM_URL

LEFT_COLOR  = (50,  220,  50)
RIGHT_COLOR = (50,  100, 255)
points: list[tuple[int, int]] = []
current_roi = "LEFT"   # "LEFT" or "RIGHT"
left_pts:  list[tuple[int, int]] = []
right_pts: list[tuple[int, int]] = []


def mouse_callback(event, x, y, flags, frame):
    global points, current_roi, left_pts, right_pts

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if current_roi == "LEFT" and len(left_pts) < 4:
        left_pts.append((x, y))
        print(f"[LEFT  {len(left_pts)}/4] ({x}, {y})")
        if len(left_pts) == 4:
            print("左車道完成，請繼續點選右車道的 4 個角點")
            current_roi = "RIGHT"

    elif current_roi == "RIGHT" and len(right_pts) < 4:
        right_pts.append((x, y))
        print(f"[RIGHT {len(right_pts)}/4] ({x}, {y})")
        if len(right_pts) == 4:
            print("\n兩個 ROI 已完成，按 Enter 輸出座標，或按 R 重設。")


def draw_state(frame: np.ndarray) -> np.ndarray:
    img = frame.copy()

    def draw_pts(pts, color, label):
        for i, p in enumerate(pts):
            cv2.circle(img, p, 6, color, -1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(pts) >= 3:
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [arr], len(pts) == 4, color, 2)
        if pts:
            cx, cy = int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))
            cv2.putText(img, label, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    draw_pts(left_pts,  LEFT_COLOR,  "LEFT")
    draw_pts(right_pts, RIGHT_COLOR, "RIGHT")

    hint = (
        f"點選左車道 {len(left_pts)}/4 角點"
        if current_roi == "LEFT"
        else f"點選右車道 {len(right_pts)}/4 角點"
    )
    cv2.putText(img, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Enter=輸出  R=重設  Q=離開",
                (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def print_result():
    print("\n" + "=" * 50)
    print("請將以下座標複製到 config.py：\n")
    print(f"LEFT_LANE_ROI = {left_pts}")
    print(f"RIGHT_LANE_ROI = {right_pts}")
    print("=" * 50 + "\n")


def main():
    global left_pts, right_pts, current_roi

    url = sys.argv[1] if len(sys.argv) > 1 else STREAM_URL
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"無法開啟影片來源：{url}")
        return

    ret, frame = cap.read()
    if not ret:
        print("無法讀取第一幀")
        cap.release()
        return
    cap.release()

    cv2.namedWindow("ROI Helper", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Helper", mouse_callback, frame)

    while True:
        display = draw_state(frame)
        cv2.imshow("ROI Helper", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == 13:   # Enter
            if len(left_pts) == 4 and len(right_pts) == 4:
                print_result()
            else:
                print("尚未完成兩個 ROI 的設定")
        elif key == ord("r"):
            left_pts.clear()
            right_pts.clear()
            current_roi = "LEFT"
            print("已重設，請重新點選")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
