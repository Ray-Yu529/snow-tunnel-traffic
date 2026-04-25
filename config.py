# ── 相機清單 ──────────────────────────────────────────────────────────────────
CAMERA_URLS = {
    "南下 (Southbound)": (
        "https://cctvn5.freeway.gov.tw/abs2mjpg/bmjpg"
        "?camera=419f9fd6-2bcf-43d0-b61f-38e081eb3eb3"
    ),
    "北上 (Northbound)": (
        "https://cctvn5.freeway.gov.tw/abs2mjpg/bmjpg"
        "?camera=1aff128d-cfce-43f4-a512-75dbe8778276"
    ),
}

# 預設方向
DEFAULT_DIRECTION = "南下 (Southbound)"
STREAM_URL = CAMERA_URLS[DEFAULT_DIRECTION]

MAX_RETRIES = 5   # 連續失敗幾次後停止
RETRY_DELAY = 3   # 失敗後等待秒數

# ── YOLOv8 設定 ───────────────────────────────────────────────────────────────
MODEL_NAME      = "yolov8n.pt"
CONF_THRESHOLD  = 0.4
TARGET_CLASSES  = ["car", "truck", "bus"]

# ── ROI 感測區（每個方向各自校正） ───────────────────────────────────────────
# 執行 python roi_helper.py 可互動式重新標記
SOUTHBOUND_LEFT_ROI  = [(7, 67), (20, 70), (196, 237), (36, 237)]
SOUTHBOUND_RIGHT_ROI = [(28, 70), (39, 68), (350, 231), (218, 236)]

# 北上待校正（暫時沿用南下座標）
NORTHBOUND_LEFT_ROI  = [(35, 49), (46, 51), (224, 234), (61, 238)]
NORTHBOUND_RIGHT_ROI = [(55, 54), (66, 54), (349, 216), (241, 236)]

ROI_BY_DIRECTION: dict[str, tuple] = {
    "南下 (Southbound)": (SOUTHBOUND_LEFT_ROI,  SOUTHBOUND_RIGHT_ROI),
    "北上 (Northbound)": (NORTHBOUND_LEFT_ROI, NORTHBOUND_RIGHT_ROI),
}

# 向下相容（traffic_analyzer 預設值用）
LEFT_LANE_ROI  = SOUTHBOUND_LEFT_ROI
RIGHT_LANE_ROI = SOUTHBOUND_RIGHT_ROI

# ── 密度與推薦穩定性設定 ──────────────────────────────────────────────────────
CONGESTED_THRESHOLD = 5    # >= 5 → 壅塞
SLOW_THRESHOLD      = 2    # >= 2 → 緩慢

DENSITY_WINDOW     = 60   # 密度滾動視窗幀數（~6 秒 @ 10fps）
REC_STABLE_WINDOW  = 50   # 推薦投票視窗幀數（~5 秒 @ 10fps）
REC_VOTE_THRESHOLD = 0.75  # 需達此比例才切換推薦（75% 多數決）

# ── 車速估算 ─────────────────────────────────────────────────────────────────
SPEED_WINDOW = 30   # 車速滾動平均幀數（~3 秒 @ 10fps）

# 相機校正：畫面縱向每公尺對應的像素數（車輛行進方向）
# 校正方式：在畫面中量出一段已知實際長度的物件（如車道虛線長 6 m）的像素高度，
#   設 PIXELS_PER_METER = 像素高度 / 6
# 初始值為估算值；觀察顯示車速是否接近雪隧速限 70 km/h 再微調。
PIXELS_PER_METER = 2.5
