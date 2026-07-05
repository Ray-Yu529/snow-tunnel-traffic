"""
透視校正工具（Homography）
────────────────────────
執行方式：
    python homography_helper.py                    # 使用 config.py 的串流 URL
    python homography_helper.py traffic_test.mp4  # 使用本機影片

操作說明：
    - 在畫面上找一段「已知實際尺寸的矩形」，例如兩條車道虛線間的路面區塊
    - 依序點選 4 個角點（建議順序：左上 → 右上 → 右下 → 左下）
    - 完成後按 Enter，於終端機輸入該矩形的實際寬度、高度（公尺）
    - 座標會印在終端機，複製貼到 config.py 的 HOMOGRAPHY_BY_DIRECTION 對應方向
    - 按 R 重設，按 Q 離開

校正後車速估算會用透視轉換算出實際位移距離，取代單一 PIXELS_PER_METER 的
平面估算，修正隧道鏡頭「近大遠小」造成的速度誤差。
"""

import sys

import cv2
import numpy as np

from config import STREAM_URL

_COLOR = (50, 220, 255)
points: list[tuple[int, int]] = []


def mouse_callback(event, x, y, flags, frame):
    if event != cv2.EVENT_LBUTTONDOWN or len(points) >= 4:
        return
    points.append((x, y))
    print(f"[{len(points)}/4] ({x}, {y})")
    if len(points) == 4:
        print("\n4 個角點已完成，按 Enter 輸入實際尺寸並輸出設定，或按 R 重設。")


def draw_state(frame: np.ndarray) -> np.ndarray:
    img = frame.copy()
    for i, p in enumerate(points):
        cv2.circle(img, p, 6, _COLOR, -1)
        cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR, 2)
    if len(points) >= 2:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], len(points) == 4, _COLOR, 2)

    hint = f"點選已知矩形角點（建議左上→右上→右下→左下） {len(points)}/4"
    cv2.putText(img, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Enter=輸出  R=重設  Q=離開",
                (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img


def print_result() -> None:
    try:
        width_m  = float(input("矩形實際寬度（公尺，點1→點2方向）："))
        height_m = float(input("矩形實際高度（公尺，點2→點3方向）："))
    except ValueError:
        print("輸入無效，請重新按 Enter 再試一次")
        return

    world_pts = [(0, 0), (width_m, 0), (width_m, height_m), (0, height_m)]
    print("\n" + "=" * 50)
    print("請將以下設定複製到 config.py 的 HOMOGRAPHY_BY_DIRECTION 對應方向：\n")
    print(f'{{"image_pts": {points}, "world_pts": {world_pts}}}')
    print("=" * 50 + "\n")


def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else STREAM_URL
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"無法開啟影片來源：{url}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("無法讀取第一幀")
        return

    cv2.namedWindow("Homography Helper", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Homography Helper", mouse_callback, frame)

    while True:
        cv2.imshow("Homography Helper", draw_state(frame))
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        elif key == 13:   # Enter
            if len(points) == 4:
                print_result()
            else:
                print("尚未點選 4 個角點")
        elif key == ord("r"):
            points.clear()
            print("已重設，請重新點選")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
