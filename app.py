import cv2
import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

import config
from traffic_analyzer import TrafficAnalyzer, load_model
from video_stream import VideoStream

st.set_page_config(
    page_title="雪隧路況分析",
    page_icon="🚇",
    layout="wide",
)

_STATUS_ICON = {"暢通": "🟢", "緩慢": "🟡", "壅塞": "🔴"}
_TYPE_ICON   = {"car": "🚗", "truck": "🚚", "bus": "🚌"}


def _format_types(types: dict) -> str:
    return "　".join(f"{_TYPE_ICON.get(name, '🚘')} {name} {count}" for name, count in types.items())


def _cleanup_session():
    """釋放串流連線並清除 session_state，供停止分析或發生錯誤時呼叫。"""
    if "stream" in st.session_state:
        st.session_state.stream.release()
    st.session_state.pop("stream", None)
    st.session_state.pop("analyzer", None)
    st.session_state.pop("direction", None)


@st.cache_resource(show_spinner="載入 YOLOv8 模型中…")
def get_model():
    """Streamlit 每次互動都會重跑腳本，快取模型避免重複載入。"""
    return load_model()

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
        f"- 壅塞：≥ {config.CONGESTED_THRESHOLD} 輛/幀\n"
        f"（車數已依 ROI 面積正規化）\n\n"
        f"**推薦穩定**\n\n"
        f"投票視窗 {config.REC_STABLE_WINDOW} 幀 · 門檻 {int(config.REC_VOTE_THRESHOLD*100)}%\n\n"
        f"**車速估算**\n\n"
        f"滾動視窗 {config.SPEED_WINDOW} 幀 · "
        f"校正係數 {config.PIXELS_PER_METER} px/m\n"
        f"若數值偏差，請修改 `config.py` 的 `PIXELS_PER_METER`\n\n"
        f"**車流量**\n\n"
        f"經過畫面中線的車輛數，換算為近 {int(config.FLOW_WINDOW_SEC)} 秒的輛/分鐘"
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
    st.subheader("🚦 車流量")
    left_flow_ph  = st.empty()
    right_flow_ph = st.empty()

    st.markdown("---")
    st.subheader("🚙 車種組成")
    left_types_ph  = st.empty()
    right_types_ph = st.empty()

    st.markdown("---")
    st.subheader("✅ 建議車道")
    rec_ph = st.empty()

    st.markdown("---")
    fps_ph    = st.empty()
    count_ph  = st.empty()

st.markdown("---")
st.subheader("📈 近 10 分鐘趨勢")
trend_col_density, trend_col_speed = st.columns(2, gap="medium")
with trend_col_density:
    st.caption("車道密度（輛/幀，已正規化）")
    trend_density_ph = st.empty()
with trend_col_speed:
    st.caption("平均車速（km/h）")
    trend_speed_ph = st.empty()

# ── 串流主迴圈 ─────────────────────────────────────────────────────────────────
if not run:
    _cleanup_session()
    frame_ph.info("👈 勾選左側「▶ 開始分析」以啟動串流分析")
    st.stop()

try:
    model = get_model()
except RuntimeError as exc:
    st.error(f"模型載入失敗：{exc}")
    st.stop()

# 只有第一次啟動或切換方向時才重建串流與分析器；其餘 UI 互動觸發的 rerun
# 會沿用既有的連線與滾動統計，避免每次互動都重新連線、歸零密度/推薦視窗。
if st.session_state.get("direction") != direction:
    if "stream" in st.session_state:
        st.session_state.stream.release()
    st.session_state.stream    = VideoStream(url=stream_url)
    st.session_state.analyzer = TrafficAnalyzer(model=model, left_roi=left_roi, right_roi=right_roi)
    st.session_state.direction = direction

stream   = st.session_state.stream
analyzer = st.session_state.analyzer
total_frames = 0

try:
    for frame, ts in stream.frames():
        annotated, metrics = analyzer.analyze(frame, timestamp=ts)
        total_frames += 1

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_ph.image(rgb, use_container_width=True)

        li = _STATUS_ICON.get(metrics["left_status"],  "⚪")
        ri = _STATUS_ICON.get(metrics["right_status"], "⚪")

        left_metric_ph.metric(
            label=f"{li} 左線 (LEFT)",
            value=f"{metrics['left_density']:.1f} 輛/幀",
            delta=f"本幀 {metrics['left_count']} 輛",
            help=f"近 {config.DENSITY_WINDOW} 幀滾動平均（已依 ROI 面積正規化）",
        )
        right_metric_ph.metric(
            label=f"{ri} 右線 (RIGHT)",
            value=f"{metrics['right_density']:.1f} 輛/幀",
            delta=f"本幀 {metrics['right_count']} 輛",
            help=f"近 {config.DENSITY_WINDOW} 幀滾動平均（已依 ROI 面積正規化）",
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

        left_flow_ph.metric(
            label="⬅ 左線車流量",
            value=f"{metrics['left_flow']:.0f} 輛/分",
            help=f"近 {int(config.FLOW_WINDOW_SEC)} 秒內經過計數線的車輛數換算",
        )
        right_flow_ph.metric(
            label="➡ 右線車流量",
            value=f"{metrics['right_flow']:.0f} 輛/分",
            help=f"近 {int(config.FLOW_WINDOW_SEC)} 秒內經過計數線的車輛數換算",
        )

        left_types_ph.caption(f"⬅ {_format_types(metrics['left_types'])}")
        right_types_ph.caption(f"➡ {_format_types(metrics['right_types'])}")

        rec  = metrics["recommendation"]
        arrow = "←" if "左" in rec else "→"
        rec_ph.markdown(
            f"## {arrow} {rec}\n\n"
            f"左線：{li} **{metrics['left_status']}**  \n"
            f"右線：{ri} **{metrics['right_status']}**"
        )

        fps_ph.caption(f"⚡ {metrics['fps']:.1f} FPS")
        count_ph.caption(f"已處理 {total_frames:,} 幀")

        hist_df = analyzer.history_df()
        if not hist_df.empty:
            trend_density_ph.line_chart(hist_df[["左線密度", "右線密度"]], height=220)
            trend_speed_ph.line_chart(hist_df[["左線車速", "右線車速"]], height=220)
        else:
            trend_density_ph.caption("蒐集趨勢資料中…")
            trend_speed_ph.caption("蒐集趨勢資料中…")

except (RerunException, StopException):
    # Streamlit 控制流例外（使用者互動觸發 rerun、或取消分析）：
    # 不清除 session_state，讓串流與統計在下一次 rerun 時延續
    raise
except Exception as exc:
    st.error(f"串流處理發生錯誤：{exc}")

# 只有在迴圈正常結束（串流重試耗盡）或發生非 Streamlit 例外時才會執行到這裡
_cleanup_session()
frame_ph.warning("串流已停止。重新勾選「▶ 開始分析」可重新啟動。")
