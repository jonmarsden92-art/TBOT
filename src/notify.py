"""
Send trade notifications via Telegram and/or email.
All credentials are optional — bot runs fine without notifications.
"""

import os
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
SENDGRID_KEY     = os.environ.get("SENDGRID_API_KEY", "")
NOTIFY_EMAIL     = os.environ.get("NOTIFY_EMAIL", "")

REPORT_FILE = Path("logs/report.json")


def load_report() -> dict:
    if not REPORT_FILE.exists():
        return {}
    with open(REPORT_FILE) as f:
        return json.load(f)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram not configured — skipping")
        return
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=10)
        log.info("✅ Telegram notification sent")
    except Exception as e:
        log.error(f"Telegram failed: {e}")


def send_email(subject: str, body: str):
    if not SENDGRID_KEY or not NOTIFY_EMAIL:
        log.info("Email not configured — skipping")
        return
    try:
        import urllib.request
        payload = json.dumps({
            "personalizations": [{"to": [{"email": NOTIFY_EMAIL}]}],
            "from":             {"email": "bot@tradingbot.app"},
            "subject":          subject,
            "content":          [{"type": "text/plain", "value": body}],
        }).encode()
        req = urllib.request.Request(
            "https://api.sendgrid.com/v3/mail/send",
            data=payload,
            headers={
                "Authorization": f"Bearer {SENDGRID_KEY}",
                "Content-Type":  "application/json",
            },
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("✅ Email notification sent")
    except Exception as e:
        log.error(f"Email failed: {e}")


def format_message(report: dict) -> str:
    acc   = report.get("account", {})
    perf  = report.get("performance", {})
    pos   = report.get("positions", [])
    trades = report.get("trades", [])

    # Recent trades (last 3)
    recent = trades[-3:] if trades else []
    trade_lines = "\n".join(
        f"  • {t.get('side','?').upper()} {t.get('qty','?')}x {t.get('symbol','?')} "
        f"({t.get('reason','signal')})"
        for t in recent
    ) or "  None this cycle"

    # Open positions
    pos_lines = "\n".join(
        f"  • {p['symbol']}: {p['unrealized_plpc']:+.1%} P&L"
        for p in pos
    ) or "  None"

    return (
        f"🤖 *Trading Bot Update*\n"
        f"_{datetime.now().strftime('%d %b %Y %H:%M UTC')}_\n\n"
        f"💰 *Portfolio:* ${acc.get('portfolio_value', 0):,.2f}\n"
        f"💵 *Cash:*      ${acc.get('cash', 0):,.2f}\n"
        f"📈 *Return:*    {perf.get('return_pct', 0):+.2f}%\n\n"
        f"📊 *Open Positions ({len(pos)}):*\n{pos_lines}\n\n"
        f"🔄 *Recent Trades:*\n{trade_lines}\n"
    )


def notify():
    report  = load_report()
    if not report:
        log.info("No report found — nothing to notify")
        return

    message = format_message(report)
    perf    = report.get("performance", {})
    subject = (
        f"Trading Bot | "
        f"${report.get('account', {}).get('portfolio_value', 0):,.2f} | "
        f"{perf.get('return_pct', 0):+.2f}% return"
    )

    send_telegram(message)
    send_email(subject, message.replace("*", "").replace("_", ""))


if __name__ == "__main__":
    notify()
