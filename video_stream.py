import time
import logging
from typing import Generator

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


class VideoStream:
    """
    讀取高公局 CCTV 的 MJPEG 持續串流（bmjpg 端點）。

    伺服器以 multipart/MJPEG 形式持續推送 JPEG frames，直到它自己關閉連線
    （通常約 30–60 秒後）。本類別透過搜尋 JPEG SOI/EOI byte marker 拆解每幀，
    斷線時自動重連，對外提供無縫的 frames() 產生器介面。
    """

    def __init__(self, url: str = STREAM_URL):
        # 清除原始 URL 裡可能帶有的舊時間戳，每次連線時重新附加
        self._base_url = url.split("&t1968")[0].split("&t=")[0]

        self._session = requests.Session()
        self._session.verify = False
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (CCTV Analyzer)"})

    # ── 私有：單次串流連線 ─────────────────────────────────────────────────────

    def _iter_one_connection(self) -> Generator[np.ndarray, None, None]:
        """
        建立一條 MJPEG HTTP 串流連線，持續解析並 yield BGR frame。
        連線關閉或發生例外時直接 return，由 frames() 負責重連。
        """
        url = f"{self._base_url}&t1968={time.time()}"
        logger.info("建立串流連線：%s", url)

        # timeout=(connect_s, read_s)：read 設為 None 表示無限等待串流推送
        with self._session.get(url, stream=True, timeout=(10, None)) as resp:
            resp.raise_for_status()
            logger.info("連線成功，開始接收 MJPEG frames")

            buf = b""
            for chunk in resp.iter_content(chunk_size=8192):
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
                        # EOI 還沒到，等下一個 chunk
                        buf = buf[start:]
                        break

                    jpg  = buf[start : end + 2]
                    buf  = buf[end + 2:]
                    arr  = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

    # ── 公開 API ───────────────────────────────────────────────────────────────

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        持續 yield BGR ndarray。
        - 伺服器關閉連線後自動重連
        - 連續失敗 MAX_RETRIES 次後停止
        - Streamlit 取消 checkbox 時，runtime 會在 yield 點中斷此產生器
        """
        failures = 0

        while True:
            try:
                for frame in self._iter_one_connection():
                    failures = 0   # 成功收到幀，重設失敗計數
                    yield frame

                # 走到這裡代表伺服器正常關閉連線，直接重連
                logger.warning("串流正常結束，重新連線…")

            except requests.exceptions.RequestException as exc:
                failures += 1
                logger.warning("串流連線失敗 (%d/%d)：%s", failures, MAX_RETRIES, exc)
                if failures >= MAX_RETRIES:
                    logger.error("超過最大重試次數，停止串流")
                    break
                logger.info("%d 秒後重試…", RETRY_DELAY)
                time.sleep(RETRY_DELAY)

    def release(self):
        self._session.close()
        logger.info("HTTP session 已關閉")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
