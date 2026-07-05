# 🚇 雪隧車道即時路況分析

即時讀取國道五號雪山隧道 CCTV 串流，透過 YOLOv8 偵測車輛，比較左右車道密度與車速，並推薦較順暢的行駛車道。

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## ✨ 功能特色

- **即時串流**：直接讀取高公局 CCTV MJPEG 串流，斷線自動重連；背景執行緒只保留最新一幀，推論跟不上時自動丟舊幀，延遲不會累積
- **車輛偵測**：YOLOv8n 模型，推論階段直接以 COCO 類別 id 過濾小客車、卡車、公車，並以原始解析度推論（不做多餘放大）
- **雙車道分析**：ROI 多邊形劃定左右車道感測區，各自計算車輛密度（依 ROI 面積正規化，公平比較）
- **車速估算**：追蹤車輛重心逐幀位移，以實際幀間時間戳換算平均車速（km/h），車道淨空後自動歸零
- **車流量統計**：ROI 內設置虛擬計數線，統計近 60 秒經過車輛數換算為輛/分鐘
- **車種組成**：即時顯示各車道小客車／卡車／公車數量
- **趨勢圖**：近 10 分鐘車道密度與車速走勢圖，一眼看出正在惡化還是紓解
- **穩定推薦**：50 幀投票機制（75% 多數決），避免建議車道頻繁跳動
- **南下 / 北上**：支援雙向鏡頭切換，ROI 分別校正；切換方向以外的 UI 互動不會重置統計或重新連線
- **透視校正（可選）**：用 homography 取代單一比例估算，修正隧道鏡頭「近大遠小」造成的車速誤差
- **停等偵測**：追蹤到同一輛車連續數秒幾乎不動即發出警示，是隧道行車安全的重要事件
- **壅塞推播（可選）**：車道持續壅塞超過設定秒數，透過 Telegram Bot 通知
- **網頁校正 ROI**：側邊欄可直接輸入座標即時儲存，不需另開視窗

---

## 🖥️ 介面預覽

| 項目 | 說明 |
|------|------|
| 頂部警示 | 偵測到停等車輛時顯示紅色警示橫幅 |
| 左側影像 | 即時標註偵測結果、ROI 區域與車流量計數線；停等車輛標紅框 |
| 右側面板 | 車道密度、平均車速、車流量、車種組成、建議車道、FPS |
| 下方趨勢圖 | 近 10 分鐘車道密度／車速走勢 |
| 側邊欄 | 方向切換、參數設定、ROI 座標查看與線上編輯 |

---

## 🛠️ 技術架構

```
高公局 CCTV (MJPEG)
        │
        ▼
  VideoStream          ← JPEG byte-marker 解析 + 自動重連
        │
        ▼
  TrafficAnalyzer      ← YOLOv8n 偵測 + ROI 計數 + 重心追蹤
        │                 + 車流量計數線 + 停等偵測 + 壅塞通知
        ▼
   Streamlit UI        ← 即時顯示影像、密度、車速、趨勢圖、推薦車道
        │
        ▼
     notifier.py       ← Telegram 推播（可選，預設停用）
```

---

## 🚀 快速開始

### 1. 安裝相依套件

```bash
pip install -r requirements.txt
```

> YOLOv8 模型（`yolov8n.pt`，約 6 MB）會在第一次執行時自動下載。

### 2. 啟動應用程式

```bash
streamlit run app.py
```

瀏覽器開啟 `http://localhost:8501`，在側邊欄選擇行駛方向後勾選「▶ 開始分析」。

### 3. 使用 Docker（可選）

```bash
docker build -t snow-tunnel-traffic .
docker run -p 8501:8501 snow-tunnel-traffic
```

映像已內含 `yolov8n.pt`，啟動不需額外下載模型。

---

## ⚙️ 設定說明

所有參數集中於 [`config.py`](config.py)。

### ROI 校正

左右車道的感測多邊形需依據鏡頭畫面手動校正，有兩種方式：

1. **互動式工具**（適合大幅調整）：
   ```bash
   python roi_helper.py
   ```
   點選畫面上的 4 個點定義左線 ROI，再點 4 個點定義右線 ROI，終端機會輸出座標，貼回 `config.py` 對應方向的變數即可。

2. **網頁直接編輯**（適合微調）：在 app.py 側邊欄展開「ROI 座標」，直接輸入座標數字並按「💾 儲存 ROI」，會立即套用並寫入 `roi_overrides.json`（優先權高於 `config.py`，刪除該檔可恢復預設值，此檔不需版控）。

### 車速校正

車速換算預設用「畫面中每公尺對應幾個像素」的平面估算：

1. 在影像中找一段已知長度的標線（如車道虛線長度約 6 m）
2. 量出該線段的像素高度
3. 設定 `PIXELS_PER_METER = 像素高度 / 6`

顯示車速若比速限（70 km/h）偏低，將 `PIXELS_PER_METER` 調小；偏高則調大。

隧道鏡頭透視效果明顯（近大遠小），單一比例在畫面遠近兩端誤差可能不小。如需更精準，執行透視校正：

```bash
python homography_helper.py
```

點選畫面中一段已知實際尺寸的矩形（如兩條車道虛線間的路面區塊）4 個角點，輸入其實際寬高（公尺），
把輸出貼到 `config.py` 的 `HOMOGRAPHY_BY_DIRECTION` 對應方向即可。校正後車速估算會自動改用透視轉換，
不需要也不會再用到 `PIXELS_PER_METER`。

### 停等偵測

追蹤到同一輛車連續 `STOPPED_DURATION_SEC`（預設 10）秒幾乎不移動（幀間位移 < `STOPPED_PIXEL_THRESHOLD`
像素）就判定為停等，畫面上以紅框標示，並在頂部顯示警示橫幅。這是像素空間的簡單啟發式判斷，不需校正也能用，
但不做透視補償，畫面遠端的判定會比近端寬鬆一些。

### 壅塞推播（Telegram）

車道持續壅塞超過 `CONGESTION_ALERT_DURATION_SEC`（預設 15）秒會發送 Telegram 通知，
同一車道 `CONGESTION_ALERT_COOLDOWN_SEC`（預設 300）秒內不重複通知。設定方式：

1. 用 [@BotFather](https://t.me/BotFather) 建立 Bot，取得 `TELEGRAM_BOT_TOKEN`
2. 傳訊息給你的 Bot 後，用 `https://api.telegram.org/bot<TOKEN>/getUpdates` 找出 `chat.id`，填入 `TELEGRAM_CHAT_ID`
3. 兩者留空（預設）則此功能停用，不影響其他功能運作

### 主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `CONF_THRESHOLD` | 0.4 | YOLOv8 信心門檻 |
| `INFER_IMGSZ` | 384 | YOLO 推論解析度（貼近來源畫面尺寸，避免無謂放大） |
| `DEVICE` | None | 推論裝置，如 `"cuda:0"`；None 交給 ultralytics 自動偵測 |
| `HALF_PRECISION` | False | GPU 支援 FP16 時可設 True 加速 |
| `SLOW_THRESHOLD` | 2 | 緩慢判定（輛/幀） |
| `CONGESTED_THRESHOLD` | 5 | 壅塞判定（輛/幀） |
| `DENSITY_WINDOW` | 60 | 密度滾動平均幀數 |
| `REC_STABLE_WINDOW` | 50 | 推薦投票視窗幀數 |
| `REC_VOTE_THRESHOLD` | 0.75 | 推薦切換門檻（75%） |
| `SPEED_WINDOW` | 30 | 車速滾動平均幀數 |
| `SPEED_STALE_SEC` | 2.0 | 車道淨空超過此秒數則車速歸零 |
| `PIXELS_PER_METER` | 2.5 | 相機校正係數（需調整） |
| `FLOW_LINE_RATIO` | 0.5 | 車流量計數線在 ROI 垂直範圍的位置比例 |
| `FLOW_WINDOW_SEC` | 60 | 車流量統計的滾動時間窗（秒） |
| `HISTORY_SAMPLE_INTERVAL_SEC` | 2 | 趨勢圖取樣間隔（秒） |
| `HISTORY_MAX_POINTS` | 300 | 趨勢圖最多保留筆數（約 10 分鐘） |
| `HOMOGRAPHY_BY_DIRECTION` | None | 透視校正矩陣設定，None 時退回 `PIXELS_PER_METER` 平面估算 |
| `STOPPED_PIXEL_THRESHOLD` | 3.0 | 幀間位移小於此像素視為「本幀未移動」 |
| `STOPPED_DURATION_SEC` | 10.0 | 連續未移動超過此秒數判定為停等 |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | "" | Telegram 推播設定，留空即停用 |
| `CONGESTION_ALERT_DURATION_SEC` | 15.0 | 持續壅塞超過此秒數才通知 |
| `CONGESTION_ALERT_COOLDOWN_SEC` | 300.0 | 同一車道兩次通知的最短間隔 |

---

## 🧪 測試

```bash
pip install -r requirements-dev.txt
pytest
```

涵蓋密度門檻判斷、推薦投票穩定性、透視/平面車速換算、壅塞通知節流、MJPEG buffer 解析等純邏輯，
不需要真的連上 CCTV 或載入 YOLO 權重。

---

## 📁 專案結構

```
snow-tunnel-traffic/
├── app.py                  # Streamlit 主程式
├── traffic_analyzer.py     # YOLOv8 偵測 + 密度/車速/車流量/停等分析
├── video_stream.py         # MJPEG 串流讀取 + 自動重連
├── notifier.py             # Telegram 推播（可選，預設停用）
├── config.py               # 所有參數、ROI 座標與網頁覆寫載入
├── roi_helper.py           # 互動式 ROI 校正工具
├── homography_helper.py    # 互動式透視校正工具
├── tests/                  # pytest 單元測試
├── Dockerfile / .dockerignore
├── requirements.txt
└── requirements-dev.txt    # 含測試用套件
```

---

## 📡 資料來源

CCTV 串流由**交通部高速公路局**提供，僅供學術研究與個人學習使用。

---

## 🗺️ 尚未實作

以下項目需要目前不具備的外部資源，暫緩實作：

- **多鏡頭全隧道視圖**：需要雪隧沿線其他斷面的 CCTV 攝影機 ID
- **與 TDX 運輸資料流通服務比對**：需要至 [tdx.transportdata.tw](https://tdx.transportdata.tw) 申請 API 金鑰

---

## 📄 License

[Apache-2.0](LICENSE)
