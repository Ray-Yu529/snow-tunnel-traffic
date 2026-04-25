import math
from collections import deque
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONF_THRESHOLD,
    CONGESTED_THRESHOLD,
    DENSITY_WINDOW,
    LEFT_LANE_ROI,
    MODEL_NAME,
    PIXELS_PER_METER,
    REC_STABLE_WINDOW,
    REC_VOTE_THRESHOLD,
    RIGHT_LANE_ROI,
    SLOW_THRESHOLD,
    SPEED_WINDOW,
    TARGET_CLASSES,
)

_LEFT_COLOR  = (50,  220,  50)
_RIGHT_COLOR = (50,  100, 255)
_OTHER_COLOR = (160, 160, 160)


def density_to_status(density: float) -> str:
    if density >= CONGESTED_THRESHOLD:
        return "壅塞"
    if density >= SLOW_THRESHOLD:
        return "緩慢"
    return "暢通"


class TrafficAnalyzer:
    """
    YOLOv8 車輛偵測 + 雙車道密度分析。

    推薦穩定機制：
      - 密度用 DENSITY_WINDOW 幀滾動平均，減少單幀噪音
      - 推薦用 REC_STABLE_WINDOW 幀投票，需達 REC_VOTE_THRESHOLD 多數才切換，
        防止左右頻繁跳動
    """

    def __init__(
        self,
        left_roi:  Optional[list] = None,
        right_roi: Optional[list] = None,
    ):
        try:
            self.model = YOLO(MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(f"無法載入模型 '{MODEL_NAME}'：{exc}") from exc

        self._target_set: set[str] = set(TARGET_CLASSES)
        self._target_ids: Optional[set[int]] = None

        self._left_roi  = np.array(left_roi  or LEFT_LANE_ROI,  dtype=np.int32)
        self._right_roi = np.array(right_roi or RIGHT_LANE_ROI, dtype=np.int32)

        # 密度滾動視窗
        self._left_buf:  deque[int] = deque(maxlen=DENSITY_WINDOW)
        self._right_buf: deque[int] = deque(maxlen=DENSITY_WINDOW)

        # 推薦穩定投票視窗（"L" or "R"）
        self._rec_votes: deque[str] = deque(maxlen=REC_STABLE_WINDOW)
        self._stable_rec: str = "L"   # 預設左線

        # 車速估算：逐幀追蹤重心位移
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._cur_centroids:  dict[int, tuple[float, float]] = {}
        self._left_speed_buf:  deque[float] = deque(maxlen=SPEED_WINDOW)
        self._right_speed_buf: deque[float] = deque(maxlen=SPEED_WINDOW)

    # ── 私有工具 ───────────────────────────────────────────────────────────────

    def _resolve_ids(self, names: dict[int, str]) -> set[int]:
        if self._target_ids is None:
            self._target_ids = {
                idx for idx, name in names.items() if name in self._target_set
            }
        return self._target_ids

    @staticmethod
    def _in_roi(cx: float, cy: float, roi: np.ndarray) -> bool:
        return cv2.pointPolygonTest(roi, (cx, cy), False) >= 0

    def _draw_rois(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self._left_roi],  _LEFT_COLOR)
        cv2.fillPoly(overlay, [self._right_roi], _RIGHT_COLOR)
        cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
        cv2.polylines(frame, [self._left_roi],  True, _LEFT_COLOR,  2)
        cv2.polylines(frame, [self._right_roi], True, _RIGHT_COLOR, 2)

        lc = tuple(self._left_roi.mean(axis=0).astype(int))
        rc = tuple(self._right_roi.mean(axis=0).astype(int))
        cv2.putText(frame, "LEFT",  lc, cv2.FONT_HERSHEY_DUPLEX, 0.85, _LEFT_COLOR,  2)
        cv2.putText(frame, "RIGHT", rc, cv2.FONT_HERSHEY_DUPLEX, 0.85, _RIGHT_COLOR, 2)

    def _vote_recommendation(self, left_density: float, right_density: float) -> str:
        """
        投票式滯後推薦：
        記錄最近 REC_STABLE_WINDOW 幀的即時偏好，
        只有當某一側達到 REC_VOTE_THRESHOLD 多數時才切換，
        否則維持現有推薦，避免頻繁跳動。
        """
        raw = "L" if left_density <= right_density else "R"
        self._rec_votes.append(raw)

        if len(self._rec_votes) < REC_STABLE_WINDOW // 2:
            # 資料不足，直接用即時值
            self._stable_rec = raw
            return self._stable_rec

        total = len(self._rec_votes)
        left_votes  = self._rec_votes.count("L")
        right_votes = total - left_votes

        if left_votes  / total >= REC_VOTE_THRESHOLD:
            self._stable_rec = "L"
        elif right_votes / total >= REC_VOTE_THRESHOLD:
            self._stable_rec = "R"
        # else: 票數未達門檻，維持現有推薦（滯後效果）

        return self._stable_rec

    # ── 公開 API ───────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray, fps: float = 10.0) -> tuple[np.ndarray, dict]:
        # 交換重心字典：上一幀的重心供本幀計算位移
        self._prev_centroids = self._cur_centroids
        self._cur_centroids  = {}

        results    = self.model.track(frame, conf=CONF_THRESHOLD, persist=True, verbose=False)[0]
        target_ids = self._resolve_ids(results.names)

        annotated = frame.copy()
        self._draw_rois(annotated)

        left_count = right_count = 0
        frame_left_speeds: list[float]  = []
        frame_right_speeds: list[float] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in target_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            conf   = float(box.conf[0])
            label  = results.names[cls_id]
            track_id = int(box.id[0]) if box.id is not None else None

            in_left  = self._in_roi(cx, cy, self._left_roi)
            in_right = self._in_roi(cx, cy, self._right_roi)

            # 計算位移速度（像素/幀）
            if track_id is not None and track_id in self._prev_centroids:
                px, py = self._prev_centroids[track_id]
                displacement = math.hypot(cx - px, cy - py)
                # 過濾重連後 ID 重複造成的異常跳躍（> 150 px/幀）
                if displacement < 150:
                    # px/幀 * 幀/秒 / (px/m) * 3.6 = km/h
                    speed = displacement * fps / PIXELS_PER_METER * 3.6
                    if in_left:
                        frame_left_speeds.append(speed)
                    elif in_right:
                        frame_right_speeds.append(speed)

            if track_id is not None:
                self._cur_centroids[track_id] = (cx, cy)

            if in_left:
                left_count += 1
                color = _LEFT_COLOR
            elif in_right:
                right_count += 1
                color = _RIGHT_COLOR
            else:
                color = _OTHER_COLOR

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.circle(annotated, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
            )

        # 用本幀平均速度更新滾動視窗
        if frame_left_speeds:
            self._left_speed_buf.append(float(np.mean(frame_left_speeds)))
        if frame_right_speeds:
            self._right_speed_buf.append(float(np.mean(frame_right_speeds)))

        self._left_buf.append(left_count)
        self._right_buf.append(right_count)

        left_density  = float(np.mean(self._left_buf))
        right_density = float(np.mean(self._right_buf))
        left_avg_speed  = float(np.mean(self._left_speed_buf))  if self._left_speed_buf  else 0.0
        right_avg_speed = float(np.mean(self._right_speed_buf)) if self._right_speed_buf else 0.0

        stable = self._vote_recommendation(left_density, right_density)
        recommendation = "← 左線 (LEFT)" if stable == "L" else "右線 (RIGHT) →"

        return annotated, {
            "left_count":      left_count,
            "right_count":     right_count,
            "left_density":    left_density,
            "right_density":   right_density,
            "left_status":     density_to_status(left_density),
            "right_status":    density_to_status(right_density),
            "left_avg_speed":  left_avg_speed,
            "right_avg_speed": right_avg_speed,
            "recommendation":  recommendation,
        }

    def reset_buffers(self):
        self._left_buf.clear()
        self._right_buf.clear()
        self._rec_votes.clear()
        self._stable_rec = "L"
        self._left_speed_buf.clear()
        self._right_speed_buf.clear()
        self._prev_centroids.clear()
        self._cur_centroids.clear()
