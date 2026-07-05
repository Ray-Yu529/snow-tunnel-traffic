from video_stream import VideoStream, _EOI, _SOI


def test_extract_jpegs_splits_multiple_frames_from_one_chunk():
    jpg1 = _SOI + b"AAA" + _EOI
    jpg2 = _SOI + b"BBB" + _EOI

    frames, remaining = VideoStream._extract_jpegs(jpg1 + jpg2)

    assert frames == [jpg1, jpg2]
    assert remaining == b""


def test_extract_jpegs_keeps_incomplete_tail_for_next_chunk():
    jpg1 = _SOI + b"AAA" + _EOI
    partial = _SOI + b"BB"   # 沒有 EOI，應該留到下一批 chunk

    frames, remaining = VideoStream._extract_jpegs(jpg1 + partial)

    assert frames == [jpg1]
    assert remaining == partial


def test_extract_jpegs_discards_junk_before_first_soi():
    jpg1 = _SOI + b"AAA" + _EOI

    frames, remaining = VideoStream._extract_jpegs(b"garbage-no-marker" + jpg1)

    assert frames == [jpg1]
    assert remaining == b""


def test_extract_jpegs_returns_empty_when_no_soi_found():
    frames, remaining = VideoStream._extract_jpegs(b"no markers here at all")

    assert frames == []
    assert remaining == b""


def test_extract_jpegs_drops_oversized_buffer_missing_eoi(monkeypatch):
    monkeypatch.setattr("video_stream._MAX_BUFFER_SIZE", 10)
    corrupted = _SOI + b"X" * 20   # 一直等不到 EOI，且超過上限

    frames, remaining = VideoStream._extract_jpegs(corrupted)

    assert frames == []
    assert remaining == b""
