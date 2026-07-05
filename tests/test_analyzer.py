import pytest

import config
from traffic_analyzer import TrafficAnalyzer, density_to_status


class DummyModel:
    """analyze()（會呼叫 YOLO 推論）不在這些測試中被使用，僅用來跳過真正的模型載入。"""


def make_analyzer(**kwargs) -> TrafficAnalyzer:
    return TrafficAnalyzer(
        model=DummyModel(),
        left_roi=config.SOUTHBOUND_LEFT_ROI,
        right_roi=config.SOUTHBOUND_RIGHT_ROI,
        **kwargs,
    )


def test_density_to_status_thresholds():
    assert density_to_status(0) == "暢通"
    assert density_to_status(config.SLOW_THRESHOLD) == "緩慢"
    assert density_to_status(config.CONGESTED_THRESHOLD) == "壅塞"


def test_vote_recommendation_uses_raw_value_before_enough_samples():
    analyzer = make_analyzer()
    result = analyzer._vote_recommendation(1.0, 5.0)   # 左線密度較低 -> "L"
    assert result == "L"


def test_vote_recommendation_ignores_single_noisy_vote_once_stable():
    analyzer = make_analyzer()
    for _ in range(config.REC_STABLE_WINDOW):
        analyzer._vote_recommendation(1.0, 5.0)   # 一路投左線
    assert analyzer._stable_rec == "L"

    # 混入一票相反的雜訊，尚不足以撼動 75% 多數，應維持原推薦
    result = analyzer._vote_recommendation(5.0, 1.0)
    assert result == "L"


def test_vote_recommendation_switches_on_sustained_majority():
    analyzer = make_analyzer()
    for _ in range(config.REC_STABLE_WINDOW):
        analyzer._vote_recommendation(5.0, 1.0)   # 一路投右線
    assert analyzer._stable_rec == "R"


def test_world_distance_flat_fallback_uses_pixels_per_meter():
    analyzer = make_analyzer()
    dist = analyzer._world_distance((0, 0), (config.PIXELS_PER_METER, 0))
    assert dist == pytest.approx(1.0)


def test_world_distance_uses_homography_when_calibrated():
    direction = "南下 (Southbound)"
    image_pts = [(0, 0), (100, 0), (100, 100), (0, 100)]
    world_pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
    config.HOMOGRAPHY_BY_DIRECTION[direction] = {"image_pts": image_pts, "world_pts": world_pts}
    try:
        analyzer = make_analyzer(direction=direction)
        dist = analyzer._world_distance((0, 0), (100, 0))
        assert dist == pytest.approx(10.0, rel=1e-3)
    finally:
        config.HOMOGRAPHY_BY_DIRECTION[direction] = None


def test_congestion_alert_requires_sustained_duration_then_cools_down(monkeypatch):
    sent = []
    monkeypatch.setattr("traffic_analyzer.send_telegram_message", sent.append)

    analyzer = make_analyzer(direction="南下 (Southbound)")

    analyzer._check_congestion("L", "壅塞", timestamp=0.0)
    assert sent == []   # 剛開始壅塞，尚未達到門檻秒數

    analyzer._check_congestion("L", "壅塞", timestamp=config.CONGESTION_ALERT_DURATION_SEC + 1)
    assert len(sent) == 1

    # 冷卻時間內即使持續壅塞也不應重複通知
    analyzer._check_congestion("L", "壅塞", timestamp=config.CONGESTION_ALERT_DURATION_SEC + 2)
    assert len(sent) == 1

    # 恢復暢通後應重置壅塞起算時間
    analyzer._check_congestion("L", "暢通", timestamp=config.CONGESTION_ALERT_DURATION_SEC + 3)
    assert analyzer._congested_since["L"] is None


def test_reset_buffers_clears_new_state():
    analyzer = make_analyzer()
    analyzer._slow_since[1] = 0.0
    analyzer._congested_since["L"] = 0.0
    analyzer._last_alert_ts["L"] = 5.0

    analyzer.reset_buffers()

    assert analyzer._slow_since == {}
    assert analyzer._congested_since == {"L": None, "R": None}
    assert analyzer._last_alert_ts == {"L": None, "R": None}
