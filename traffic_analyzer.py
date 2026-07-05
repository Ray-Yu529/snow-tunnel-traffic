import math
from collections import deque
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from config import (
    CONF_THRESHOLD,
    CONGESTED_THRESHOLD,
    DENSITY_WINDOW,
    DEVICE,
    FLOW_LINE_RATIO,
    FLOW_WINDOW_SEC,
    HALF_PRECISION,
    HISTORY_MAX_POINTS,
    HISTORY_SAMPLE_INTERVAL_SEC,
    INFER_IMGSZ,
    LEFT_LANE_ROI,
    MODEL_NAME,
    PIXELS_PER_METER,
    REC_STABLE_WINDOW,
    REC_VOTE_THRESHOLD,
    RIGHT_LANE_ROI,
    SLOW_THRESHOLD,
    SPEED_STALE_SEC,
    SPEED_WINDOW,
    TARGET_CLASS_IDS,
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


def load_model() -> YOLO:
    """載入 YOLO 模型。app.py 以 st.cache_resource 快取此結果，避免每次重跑都重新載入。"""
    try:
        return YOLO(MODEL_NAME)
    except Exception as exc:
        raise RuntimeError(f"無法載入模型 '{MODEL_NAME}'：{exc}") from exc


class TrafficAnalyzer:
    """
    YOLOv8 車輛偵測 + 雙車道密度分析。

    推薦穩定機制：
      - 密度用 DENSITY_WINDOW 幀滾動平均，減少單幀噪音
      - 密度依 ROI 面積正規化，避免面積大的車道天生數到較多車
      - 推薦用 REC_STABLE_WINDOW 幀投票，需達 REC_VOTE_THRESHOLD 多數才切換，
        防止左右頻繁跳動
    """

    def __init__(
        self,
        left_roi:  Optional[list] = None,
        right_roi: Optional[list] = None,
        model:     Optional[YOLO] = None,
    ):
        self.model = model if model is not None else load_model()

        self._left_roi  = np.array(left_roi  or LEFT_LANE_ROI,  dtype=np.int32)
        self._right_roi = np.array(right_roi or RIGHT_LANE_ROI, dtype=np.int32)

        # 面積正規化：兩個 ROI 面積不同，直接比較原始車數會偏向面積大的一側。
        # 將各車道車數換算成「平均 ROI 面積下的等效車數」，密度門檻的尺度維持不變。
        left_area  = max(cv2.contourArea(self._left_roi),  1.0)
        right_area = max(cv2.contourArea(self._right_roi), 1.0)
        mean_area  = (left_area + right_area) / 2.0
        self._left_scale  = mean_area / left_area
        self._right_scale = mean_area / right_area

        # 車流量虛擬計數線：落在各 ROI 垂直範圍的 FLOW_LINE_RATIO 比例位置
        self._left_line_y  = self._line_y(self._left_roi)
        self._right_line_y = self._line_y(self._right_roi)
        self._left_line_x  = (int(self._left_roi[:,  0].min()), int(self._left_roi[:,  0].max()))
        self._right_line_x = (int(self._right_roi[:, 0].min()), int(self._right_roi[:, 0].max()))
        self._left_crossings:  deque[float] = deque()   # 通過計數線的時間戳（滾動窗內）
        self._right_crossings: deque[float] = deque()

        # 密度滾動視窗（存放面積正規化後的等效車數）
        self._left_buf:  deque[float] = deque(maxlen=DENSITY_WINDOW)
        self._right_buf: deque[float] = deque(maxlen=DENSITY_WINDOW)

        # 推薦穩定投票視窗（"L" or "R"）
        self._rec_votes: deque[str] = deque(maxlen=REC_STABLE_WINDOW)
        self._stable_rec: Optional[str] = None   # 累積足夠票數前維持 None，由第一次投票決定初始值

        # 車速估算：逐幀追蹤重心位移，改用實際幀間時間戳計算，不受處理耗時影響
        self._prev_centroids: dict[int, tuple[float, float]] = {}
        self._cur_centroids:  dict[int, tuple[float, float]] = {}
        self._left_speed_buf:  deque[float] = deque(maxlen=SPEED_WINDOW)
        self._right_speed_buf: deque[float] = deque(maxlen=SPEED_WINDOW)
        self._left_speed_last_ts:  float = 0.0
        self._right_speed_last_ts: float = 0.0
        self._prev_ts: Optional[float] = None

        # 趨勢歷史（供 UI 畫密度/車速趨勢圖），每 HISTORY_SAMPLE_INTERVAL_SEC 秒取樣一次
        self._history: deque[dict] = deque(maxlen=HISTORY_MAX_POINTS)
        self._last_history_ts: float = 0.0
        self._start_ts: Optional[float] = None

    @staticmethod
    def _line_y(roi: np.ndarray) -> int:
        y_min, y_max = int(roi[:, 1].min()), int(roi[:, 1].max())
        return int(y_min + FLOW_LINE_RATIO * (y_max - y_min))

    # ── 私有工具 ───────────────────────────────────────────────────────────────

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

        # 車流量虛擬計數線
        cv2.line(frame, (self._left_line_x[0],  self._left_line_y),
                         (self._left_line_x[1],  self._left_line_y),  _LEFT_COLOR,  1, cv2.LINE_AA)
        cv2.line(frame, (self._right_line_x[0], self._right_line_y),
                         (self._right_line_x[1], self._right_line_y), _RIGHT_COLOR, 1, cv2.LINE_AA)

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
            # 資料不足：只在尚未有任何推薦時才用即時值初始化，
            # 避免啟動初期樣本太少導致推薦頻繁跳動
            if self._stable_rec is None:
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

    def analyze(self, frame: np.ndarray, timestamp: float) -> tuple[np.ndarray, dict]:
        """
        timestamp: 該幀實際到達時間（perf_counter，來自 VideoStream），
        用於計算幀間時間差以估算車速，不受本函式處理耗時影響。
        """
        # 交換重心字典：上一幀的重心供本幀計算位移
        self._prev_centroids = self._cur_centroids
        self._cur_centroids  = {}

        dt = timestamp - self._prev_ts if self._prev_ts is not None else 0.1
        dt = max(dt, 1e-3)
        self._prev_ts = timestamp
        if self._start_ts is None:
            self._start_ts = timestamp

        track_kwargs = dict(
            conf=CONF_THRESHOLD, imgsz=INFER_IMGSZ, classes=TARGET_CLASS_IDS,
            persist=True, verbose=False,
        )
        if DEVICE is not None:
            track_kwargs["device"] = DEVICE
        if HALF_PRECISION:
            track_kwargs["half"] = True
        results = self.model.track(frame, **track_kwargs)[0]

        annotated = frame.copy()
        self._draw_rois(annotated)

        left_count = right_count = 0
        left_types  = dict.fromkeys(TARGET_CLASSES, 0)
        right_types = dict.fromkeys(TARGET_CLASSES, 0)
        frame_left_speeds: list[float]  = []
        frame_right_speeds: list[float] = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            conf   = float(box.conf[0])
            label  = results.names[cls_id]
            track_id = int(box.id[0]) if box.id is not None else None

            in_left  = self._in_roi(cx, cy, self._left_roi)
            in_right = self._in_roi(cx, cy, self._right_roi)

            # 計算位移速度，並偵測是否跨越車流量計數線
            if track_id is not None and track_id in self._prev_centroids:
                px, py = self._prev_centroids[track_id]
                displacement = math.hypot(cx - px, cy - py)
                # 過濾重連後 ID 重複造成的異常跳躍（> 150 px/幀）
                if displacement < 150:
                    # px / 秒 / (px/m) * 3.6 = km/h
                    speed = displacement / dt / PIXELS_PER_METER * 3.6
                    if in_left:
                        frame_left_speeds.append(speed)
                    elif in_right:
                        frame_right_speeds.append(speed)

                    if in_left and (py - self._left_line_y) * (cy - self._left_line_y) < 0:
                        self._left_crossings.append(timestamp)
                    elif in_right and (py - self._right_line_y) * (cy - self._right_line_y) < 0:
                        self._right_crossings.append(timestamp)

            if track_id is not None:
                self._cur_centroids[track_id] = (cx, cy)

            if in_left:
                left_count += 1
                left_types[label] = left_types.get(label, 0) + 1
                color = _LEFT_COLOR
            elif in_right:
                right_count += 1
                right_types[label] = right_types.get(label, 0) + 1
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

        # 用本幀平均速度更新滾動視窗；車道淨空超過 SPEED_STALE_SEC 秒則清空，
        # 避免畫面卡在最後一筆舊車速
        if frame_left_speeds:
            self._left_speed_buf.append(float(np.mean(frame_left_speeds)))
            self._left_speed_last_ts = timestamp
        elif timestamp - self._left_speed_last_ts > SPEED_STALE_SEC:
            self._left_speed_buf.clear()

        if frame_right_speeds:
            self._right_speed_buf.append(float(np.mean(frame_right_speeds)))
            self._right_speed_last_ts = timestamp
        elif timestamp - self._right_speed_last_ts > SPEED_STALE_SEC:
            self._right_speed_buf.clear()

        self._left_buf.append(left_count * self._left_scale)
        self._right_buf.append(right_count * self._right_scale)

        left_density  = float(np.mean(self._left_buf))
        right_density = float(np.mean(self._right_buf))
        left_avg_speed  = float(np.mean(self._left_speed_buf))  if self._left_speed_buf  else 0.0
        right_avg_speed = float(np.mean(self._right_speed_buf)) if self._right_speed_buf else 0.0

        # 車流量：清掉滾動窗外的計數，窗長 = FLOW_WINDOW_SEC 秒
        while self._left_crossings and self._left_crossings[0] < timestamp - FLOW_WINDOW_SEC:
            self._left_crossings.popleft()
        while self._right_crossings and self._right_crossings[0] < timestamp - FLOW_WINDOW_SEC:
            self._right_crossings.popleft()
        left_flow  = len(self._left_crossings)  * (60.0 / FLOW_WINDOW_SEC)
        right_flow = len(self._right_crossings) * (60.0 / FLOW_WINDOW_SEC)

        stable = self._vote_recommendation(left_density, right_density)
        recommendation = "← 左線 (LEFT)" if stable == "L" else "右線 (RIGHT) →"

        # 趨勢歷史取樣
        if timestamp - self._last_history_ts >= HISTORY_SAMPLE_INTERVAL_SEC:
            self._history.append({
                "經過分鐘": round((timestamp - self._start_ts) / 60, 2),
                "左線密度": left_density,
                "右線密度": right_density,
                "左線車速": left_avg_speed,
                "右線車速": right_avg_speed,
            })
            self._last_history_ts = timestamp

        return annotated, {
            "fps":             1.0 / dt,
            "left_count":      left_count,
            "right_count":     right_count,
            "left_density":    left_density,
            "right_density":   right_density,
            "left_status":     density_to_status(left_density),
            "right_status":    density_to_status(right_density),
            "left_avg_speed":  left_avg_speed,
            "right_avg_speed": right_avg_speed,
            "left_types":      left_types,
            "right_types":     right_types,
            "left_flow":       left_flow,
            "right_flow":      right_flow,
            "recommendation":  recommendation,
        }

    def history_df(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame(
                columns=["經過分鐘", "左線密度", "右線密度", "左線車速", "右線車速"]
            ).set_index("經過分鐘")
        return pd.DataFrame(self._history).set_index("經過分鐘")

    def reset_buffers(self):
        self._left_buf.clear()
        self._right_buf.clear()
        self._rec_votes.clear()
        self._stable_rec = None
        self._left_speed_buf.clear()
        self._right_speed_buf.clear()
        self._left_speed_last_ts = 0.0
        self._right_speed_last_ts = 0.0
        self._prev_centroids.clear()
        self._cur_centroids.clear()
        self._left_crossings.clear()
        self._right_crossings.clear()
        self._prev_ts = None
        self._history.clear()
        self._last_history_ts = 0.0
        self._start_ts = None
