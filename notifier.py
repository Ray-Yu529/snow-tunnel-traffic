import logging

import requests

import config

logger = logging.getLogger(__name__)


def send_telegram_message(text: str) -> None:
    """發送 Telegram 通知；未設定 TELEGRAM_BOT_TOKEN/CHAT_ID 時直接略過（功能預設停用）。"""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(
            url,
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text},
            timeout=5,
        )
    except requests.exceptions.RequestException as exc:
        logger.warning("Telegram 推播失敗：%s", exc)
