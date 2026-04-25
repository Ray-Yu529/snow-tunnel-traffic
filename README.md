# 🚇 雪隧車道即時路況分析

即時讀取國道五號雪山隧道 CCTV 串流，透過 YOLOv8 偵測車輛，比較左右車道密度與車速，並推薦較順暢的行駛車道。

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

---

## ✨ 功能特色

- **即時串流**：直接讀取高公局 CCTV MJPEG 串流，斷線自動重連
- **車輛偵測**：YOLOv8n 模型，過濾小客車、卡車、公車
- **雙車道分析**：ROI 多邊形劃定左右車道感測區，各自計算車輛密度
- **車速估算**：追蹤車輛重心逐幀位移，換算平均車速（km/h）
- **穩定推薦**：50 幀投票機制（75% 多數決），避免建議車道頻繁跳動
- **南下 / 北上**：支援雙向鏡頭切換，ROI 分別校正

---

## 🖥️ 介面預覽

| 項目 | 說明 |
|------|------|
| 左側影像 | 即時標註偵測結果與 ROI 區域 |
| 右側面板 | 車道密度、平均車速、建議車道、FPS |
| 側邊欄 | 方向切換、參數設定、ROI 座標查看 |

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
        │
        ▼
   Streamlit UI        ← 即時顯示影像、密度、車速、推薦車道
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

---

## ⚙️ 設定說明

所有參數集中於 [`config.py`](config.py)。

### ROI 校正

左右車道的感測多邊形需依據鏡頭畫面手動校正：

```bash
python roi_helper.py
```

點選畫面上的 4 個點定義左線 ROI，再點 4 個點定義右線 ROI，終端機會輸出座標，貼回 `config.py` 對應方向的變數即可。

### 車速校正

車速換算需要知道「畫面中每公尺對應幾個像素」：

1. 在影像中找一段已知長度的標線（如車道虛線長度約 6 m）
2. 量出該線段的像素高度
3. 設定 `PIXELS_PER_METER = 像素高度 / 6`

顯示車速若比速限（70 km/h）偏低，將 `PIXELS_PER_METER` 調小；偏高則調大。

### 主要參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `CONF_THRESHOLD` | 0.4 | YOLOv8 信心門檻 |
| `SLOW_THRESHOLD` | 2 | 緩慢判定（輛/幀） |
| `CONGESTED_THRESHOLD` | 5 | 壅塞判定（輛/幀） |
| `DENSITY_WINDOW` | 60 | 密度滾動平均幀數 |
| `REC_STABLE_WINDOW` | 50 | 推薦投票視窗幀數 |
| `REC_VOTE_THRESHOLD` | 0.75 | 推薦切換門檻（75%） |
| `SPEED_WINDOW` | 30 | 車速滾動平均幀數 |
| `PIXELS_PER_METER` | 2.5 | 相機校正係數（需調整） |

---

## 📁 專案結構

```
snow-tunnel-traffic/
├── app.py               # Streamlit 主程式
├── traffic_analyzer.py  # YOLOv8 偵測 + 密度 + 車速分析
├── video_stream.py      # MJPEG 串流讀取 + 自動重連
├── config.py            # 所有參數與 ROI 座標
├── roi_helper.py        # 互動式 ROI 校正工具
└── requirements.txt
```

---

## 📡 資料來源

CCTV 串流由**交通部高速公路局**提供，僅供學術研究與個人學習使用。

---

## 📄 License

[Apache-2.0](LICENSE)
