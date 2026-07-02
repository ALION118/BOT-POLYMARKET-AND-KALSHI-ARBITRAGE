"""
Simple Telegram notifier for arbitrage opportunities.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment variables
(loaded from .env via python-dotenv, same pattern the rest of the project uses).

Fails silently (just logs a warning) if not configured, so the bot keeps
working normally even if you haven't set up Telegram yet.
"""
import os
import logging
import httpx

logger = logging.getLogger(__name__)

_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
_ENABLED = bool(_BOT_TOKEN and _CHAT_ID)

if not _ENABLED:
    logger.warning(
        "Telegram notifications disabled — set TELEGRAM_BOT_TOKEN and "
        "TELEGRAM_CHAT_ID in your .env to enable them."
    )


async def send_telegram_message(text: str) -> None:
    """Send a message to the configured Telegram chat. No-op if not configured."""
    if not _ENABLED:
        return

    url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": _CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except Exception as e:
        # Never let a notification failure crash the bot
        logger.warning(f"Failed to send Telegram notification: {e}")


async def notify_opportunity(
    opportunity_type: str,
    market_id: str,
    edge: float,
    suggested_size: float,
) -> None:
    """Formatted notification for a detected arbitrage opportunity."""
    text = (
        f"🚨 *Oportunidad de arbitraje detectada*\n\n"
        f"Tipo: `{opportunity_type}`\n"
        f"Mercado: `{market_id}`\n"
        f"Edge: `{edge:.2%}`\n"
        f"Tamaño sugerido: `${suggested_size:.2f}`"
    )
    await send_telegram_message(text)


async def notify_cross_platform_opportunity(
    question: str,
    token: str,
    buy_platform: str,
    buy_price: float,
    sell_platform: str,
    sell_price: float,
    edge_pct: float,
    suggested_size: float,
) -> None:
    """
    Formatted notification for a cross-platform (Polymarket vs Kalshi)
    arbitrage opportunity. Tells you exactly what to do on each platform.
    """
    text = (
        f"💰 *Arbitraje CROSS-PLATFORM detectado* ({edge_pct:.2%} edge)\n\n"
        f"Mercado: _{question}_\n\n"
        f"1️⃣ COMPRA `{token}` en *{buy_platform.upper()}* @ `${buy_price:.3f}`\n"
        f"2️⃣ VENDE `{token}` en *{sell_platform.upper()}* @ `${sell_price:.3f}`\n\n"
        f"Tamaño sugerido: `${suggested_size:.2f}`\n"
        f"Modo: DRY RUN (simulado, sin dinero real)"
    )
    await send_telegram_message(text)
