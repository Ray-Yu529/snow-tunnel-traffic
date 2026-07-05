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
# COCO 類別 id：2=car, 5=bus, 7=truck（推論階段直接過濾，減少後處理與雜訊）
TARGET_CLASS_IDS = [2, 5, 7]

# 推論解析度；來源 CCTV 畫面約 352x240，維持原尺寸可避免 YOLO 預設放大到 640x640
# 造成的多餘運算與畫質模糊
INFER_IMGSZ = 384

# 推論裝置："cpu"、"cuda:0" 等；None 表示交給 ultralytics 自動偵測
DEVICE     = None
HALF_PRECISION = False   # 有支援 FP16 的 GPU 時可設 True 加速

# ── ROI 感測區（每個方向各自校正） ───────────────────────────────────────────
# 執行 python roi_helper.py 可互動式重新標記
SOUTHBOUND_LEFT_ROI  = [(7, 67), (20, 70), (196, 237), (36, 237)]
SOUTHBOUND_RIGHT_ROI = [(28, 70), (39, 68), (350, 231), (218, 236)]

# 北上車道 ROI（已依北上鏡頭畫面校正，座標與南下不同）
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

# 車道淨空後，超過此秒數沒有新的車速樣本就清空滾動平均，避免畫面卡在最後一筆舊車速
SPEED_STALE_SEC = 2.0

# 相機校正：畫面縱向每公尺對應的像素數（車輛行進方向）
# 校正方式：在畫面中量出一段已知實際長度的物件（如車道虛線長 6 m）的像素高度，
#   設 PIXELS_PER_METER = 像素高度 / 6
# 初始值為估算值；觀察顯示車速是否接近雪隧速限 70 km/h 再微調。
PIXELS_PER_METER = 2.5

# ── 車流量計數（虛擬計數線） ──────────────────────────────────────────────────
# 計數線落在各 ROI 垂直範圍的比例位置（0=ROI 最上緣，1=最下緣）
FLOW_LINE_RATIO = 0.5
# 車流量統計的滾動時間窗（秒）；窗長設 60 秒可直接以「窗內計數」當作「輛/分鐘」
FLOW_WINDOW_SEC = 60.0

# ── 趨勢歷史（供 UI 畫密度/車速趨勢圖） ───────────────────────────────────────
HISTORY_SAMPLE_INTERVAL_SEC = 2    # 每隔幾秒取樣一次，避免逐幀記錄佔用過多記憶體
HISTORY_MAX_POINTS          = 300  # 300 筆 * 2 秒 = 近 10 分鐘
