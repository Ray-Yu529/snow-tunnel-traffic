import time

import cv2
import streamlit as st

import config
from traffic_analyzer import TrafficAnalyzer, density_to_status
from video_stream import VideoStream

st.set_page_config(
    page_title="雪隧路況分析",
    page_icon="🚇",
    layout="wide",
)

_STATUS_ICON = {"暢通": "🟢", "緩慢": "🟡", "壅塞": "🔴"}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")

    direction = st.radio(
        "行駛方向",
        options=list(config.CAMERA_URLS.keys()),
        index=0,
        horizontal=True,
    )
    stream_url          = config.CAMERA_URLS[direction]
    left_roi, right_roi = config.ROI_BY_DIRECTION[direction]

    st.markdown("---")
    run = st.checkbox("▶ 開始分析", value=False)

    st.markdown("---")
    with st.expander("ROI 座標"):
        st.json({"LEFT": left_roi, "RIGHT": right_roi})
        st.caption("執行 `python roi_helper.py` 重新校正")

    st.markdown("---")
    st.caption(
        f"**密度門檻**\n\n"
        f"- 暢通：< {config.SLOW_THRESHOLD} 輛/幀\n"
        f"- 緩慢：{config.SLOW_THRESHOLD}–{config.CONGESTED_THRESHOLD - 1} 輛/幀\n"
        f"- 壅塞：≥ {config.CONGESTED_THRESHOLD} 輛/幀\n\n"
        f"**推薦穩定**\n\n"
        f"投票視窗 {config.REC_STABLE_WINDOW} 幀 · 門檻 {int(config.REC_VOTE_THRESHOLD*100)}%\n\n"
        f"**車速估算**\n\n"
        f"滾動視窗 {config.SPEED_WINDOW} 幀 · "
        f"校正係數 {config.PIXELS_PER_METER} px/m\n"
        f"若數值偏差，請修改 `config.py` 的 `PIXELS_PER_METER`"
    )

# ── 主版面 ─────────────────────────────────────────────────────────────────────
st.title("🚇 雪隧車道即時路況分析")
dir_label = "⬇ 南下" if "南下" in direction else "⬆ 北上"
st.caption(f"{dir_label} · YOLOv8 車輛偵測 · 雙車道密度比較")

col_video, col_panel = st.columns([3, 1], gap="medium")

with col_video:
    frame_ph = st.empty()

with col_panel:
    st.subheader("📊 車道密度")
    left_metric_ph  = st.empty()
    right_metric_ph = st.empty()

    st.markdown("---")
    st.subheader("🏎️ 平均車速")
    left_speed_ph  = st.empty()
    right_speed_ph = st.empty()

    st.markdown("---")
    st.subheader("✅ 建議車道")
    rec_ph = st.empty()

    st.markdown("---")
    fps_ph    = st.empty()
    count_ph  = st.empty()

# ── 串流主迴圈 ─────────────────────────────────────────────────────────────────
if not run:
    frame_ph.info("👈 勾選左側「▶ 開始分析」以啟動串流分析")
    st.stop()

with st.spinner("載入 YOLOv8 模型中…"):
    try:
        analyzer = TrafficAnalyzer(left_roi=left_roi, right_roi=right_roi)
    except RuntimeError as exc:
        st.error(f"模型載入失敗：{exc}")
        st.stop()

stream       = VideoStream(url=stream_url)
total_frames = 0
prev_t       = time.perf_counter()

try:
    for frame in stream.frames():
        now    = time.perf_counter()
        fps    = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now

        annotated, metrics = analyzer.analyze(frame, fps=fps)
        total_frames += 1

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_ph.image(rgb, use_container_width=True)

        li = _STATUS_ICON.get(metrics["left_status"],  "⚪")
        ri = _STATUS_ICON.get(metrics["right_status"], "⚪")

        left_metric_ph.metric(
            label=f"{li} 左線 (LEFT)",
            value=f"{metrics['left_density']:.1f} 輛/幀",
            delta=f"本幀 {metrics['left_count']} 輛",
            help=f"近 {config.DENSITY_WINDOW} 幀滾動平均",
        )
        right_metric_ph.metric(
            label=f"{ri} 右線 (RIGHT)",
            value=f"{metrics['right_density']:.1f} 輛/幀",
            delta=f"本幀 {metrics['right_count']} 輛",
            help=f"近 {config.DENSITY_WINDOW} 幀滾動平均",
        )

        left_speed_ph.metric(
            label="⬅ 左線速度",
            value=f"{metrics['left_avg_speed']:.1f} km/h",
            help=f"近 {config.SPEED_WINDOW} 幀追蹤重心位移平均（校正係數 {config.PIXELS_PER_METER} px/m）",
        )
        right_speed_ph.metric(
            label="➡ 右線速度",
            value=f"{metrics['right_avg_speed']:.1f} km/h",
            help=f"近 {config.SPEED_WINDOW} 幀追蹤重心位移平均（校正係數 {config.PIXELS_PER_METER} px/m）",
        )

        rec  = metrics["recommendation"]
        arrow = "←" if "左" in rec else "→"
        rec_ph.markdown(
            f"## {arrow} {rec}\n\n"
            f"左線：{li} **{metrics['left_status']}**  \n"
            f"右線：{ri} **{metrics['right_status']}**"
        )

        fps_ph.caption(f"⚡ {fps:.1f} FPS")
        count_ph.caption(f"已處理 {total_frames:,} 幀")

except Exception as exc:
    st.error(f"串流處理發生錯誤：{exc}")

finally:
    stream.release()
    frame_ph.warning("串流已停止。重新勾選「▶ 開始分析」可重新啟動。")
