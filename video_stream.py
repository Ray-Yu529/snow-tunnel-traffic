import threading
import time
import logging
from typing import Generator, Optional

import urllib3
import cv2
import numpy as np
import requests

from config import MAX_RETRIES, RETRY_DELAY, STREAM_URL

# 高公局 SSL 憑證缺少 Subject Key Identifier，跳過驗證
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# JPEG 起始與結束 marker
_SOI = b"\xff\xd8"
_EOI = b"\xff\xd9"

# 串流正常時每秒推送多幀；超過此秒數收不到任何資料視為連線異常，交由重連處理
_READ_TIMEOUT = 15

# 收到 SOI 卻一直等不到 EOI（資料損毀）時，buffer 無上限成長的保護上限
_MAX_BUFFER_SIZE = 2_000_000


class VideoStream:
    """
    讀取高公局 CCTV 的 MJPEG 持續串流（bmjpg 端點）。

    伺服器以 multipart/MJPEG 形式持續推送 JPEG frames，直到它自己關閉連線
    （通常約 30–60 秒後）。本類別透過搜尋 JPEG SOI/EOI byte marker 拆解每幀，
    斷線時自動重連。

    接收工作在背景執行緒進行，且只保留「最新一幀」：
    當下游（YOLO 推論）速度跟不上串流推送速度時，自動略過來不及處理的舊幀，
    避免幀堆積在 socket 緩衝區造成顯示延遲無限累積。
    """

    def __init__(self, url: str = STREAM_URL):
        # 清除原始 URL 裡可能帶有的舊時間戳，每次連線時重新附加
        self._base_url = url.split("&t1968")[0].split("&t=")[0]

        self._session = requests.Session()
        self._session.verify = False
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (CCTV Analyzer)"})

        self._latest: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0            # 該幀解碼完成的時間戳（perf_counter）
        self._frame_ready = threading.Event()   # 有新幀可取
        self._stop        = threading.Event()   # 要求結束（release 或重試耗盡）
        self._thread: Optional[threading.Thread] = None

    # ── 私有：單次串流連線 ─────────────────────────────────────────────────────

    def _iter_one_connection(self) -> Generator[np.ndarray, None, None]:
        """
        建立一條 MJPEG HTTP 串流連線，持續解析並 yield BGR frame。
        連線關閉或發生例外時直接 return，由 _reader_loop() 負責重連。
        """
        url = f"{self._base_url}&t1968={time.time()}"
        logger.info("建立串流連線：%s", url)

        with self._session.get(url, stream=True, timeout=(10, _READ_TIMEOUT)) as resp:
            resp.raise_for_status()
            logger.info("連線成功，開始接收 MJPEG frames")

            buf = b""
            for chunk in resp.iter_content(chunk_size=8192):
                if self._stop.is_set():
                    return
                if not chunk:
                    continue
                buf += chunk

                # 從 buffer 中擷取所有完整的 JPEG（可能一次收到多幀）
                while True:
                    start = buf.find(_SOI)
                    if start == -1:
                        buf = b""   # 沒有 SOI，丟棄殘留資料
                        break

                    end = buf.find(_EOI, start + 2)
                    if end == -1:
                        # EOI 還沒到，等下一個 chunk；資料損毀導致一直等不到 EOI 時，
                        # buffer 會無上限成長，超過上限直接丟棄重新等待下一個 SOI
                        buf = buf[start:]
                        if len(buf) > _MAX_BUFFER_SIZE:
                            logger.warning("buffer 超過 %d bytes 仍未收到 EOI，捨棄殘留資料", _MAX_BUFFER_SIZE)
                            buf = b""
                        break

                    jpg  = buf[start : end + 2]
                    buf  = buf[end + 2:]
                    arr  = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

    # ── 私有：背景接收執行緒 ───────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """持續接收 frames 並覆寫 self._latest，斷線自動重連。"""
        failures = 0
        try:
            while not self._stop.is_set():
                try:
                    for frame in self._iter_one_connection():
                        failures = 0   # 成功收到幀，重設失敗計數
                        self._latest    = frame
                        self._latest_ts = time.perf_counter()
                        self._frame_ready.set()

                    if self._stop.is_set():
                        return
                    # 走到這裡代表伺服器正常關閉連線，直接重連
                    logger.warning("串流正常結束，重新連線…")

                except requests.exceptions.RequestException as exc:
                    if self._stop.is_set():
                        return   # release() 關閉 session 造成的中斷，屬正常結束
                    failures += 1
                    logger.warning("串流連線失敗 (%d/%d)：%s", failures, MAX_RETRIES, exc)
                    if failures >= MAX_RETRIES:
                        logger.error("超過最大重試次數，停止串流")
                        return
                    logger.info("%d 秒後重試…", RETRY_DELAY)
                    self._stop.wait(RETRY_DELAY)
        finally:
            # 不論何種原因結束，都喚醒 consumer 讓 frames() 能正常返回
            self._stop.set()
            self._frame_ready.set()

    # ── 公開 API ───────────────────────────────────────────────────────────────

    def frames(self) -> Generator[tuple[np.ndarray, float], None, None]:
        """
        持續 yield (最新的 BGR ndarray, 該幀解碼完成時的 perf_counter 時間戳)。
        時間戳取自背景執行緒實際收到幀的當下，不受下游處理耗時影響，
        供車速估算計算精確的幀間時間差。來不及處理的舊幀自動略過。
        - 伺服器關閉連線後由背景執行緒自動重連
        - 連續失敗 MAX_RETRIES 次後停止
        - Streamlit 取消 checkbox 時，runtime 會在 yield 點中斷此產生器
        """
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._reader_loop, name="mjpeg-reader", daemon=True
            )
            self._thread.start()

        while True:
            self._frame_ready.wait()
            if self._stop.is_set():
                return
            self._frame_ready.clear()
            frame = self._latest
            ts    = self._latest_ts
            if frame is not None:
                yield frame, ts

    def release(self):
        self._stop.set()
        self._frame_ready.set()
        self._session.close()   # 中斷進行中的連線，讓背景執行緒盡快退出
        if self._thread is not None:
            self._thread.join(timeout=2)
        logger.info("串流已關閉")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
