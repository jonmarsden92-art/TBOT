name: 🤖 Trading Bot

on:
  schedule:
    # Every 30 mins from 9:30 AM to 4:00 PM ET (14:30-21:00 UTC), Mon-Fri
    - cron: "30 14 * * 1-5"
    - cron: "00 15 * * 1-5"
    - cron: "30 15 * * 1-5"
    - cron: "00 16 * * 1-5"
    - cron: "30 16 * * 1-5"
    - cron: "00 17 * * 1-5"
    - cron: "30 17 * * 1-5"
    - cron: "00 18 * * 1-5"
    - cron: "30 18 * * 1-5"
    - cron: "00 19 * * 1-5"
    - cron: "30 19 * * 1-5"
    - cron: "00 20 * * 1-5"
    - cron: "30 20 * * 1-5"
    - cron: "55 20 * * 1-5"

  workflow_dispatch:

jobs:
  trade:
    name: Run Trading Bot
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Dependencies
        run: |
          pip install --upgrade pip
          pip install alpaca-trade-api==3.2.0 yfinance pandas numpy requests websocket-client

      - name: Create Logs Directory
        run: mkdir -p logs

      - name: Restore Bot State
        uses: actions/cache@v4
        with:
          path: logs/
          key: bot-state-${{ github.run_number }}
          restore-keys: |
            bot-state-

      - name: Run Trading Bot
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
          ALPACA_BASE_URL: ${{ secrets.ALPACA_BASE_URL }}
        run: python src/bot.py

      - name: Save Bot State
        uses: actions/cache/save@v4
        if: always()
        with:
          path: logs/
          key: bot-state-${{ github.run_number }}

      - name: Upload Logs as Artifact
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: bot-logs-${{ github.run_number }}
          path: logs/
          retention-days: 30

      - name: Generate Performance Report
        if: always()
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
          ALPACA_BASE_URL: ${{ secrets.ALPACA_BASE_URL }}
        run: python src/report.py

      - name: Commit Report to Repo
        if: always()
        run: |
          git config user.name "Trading Bot"
          git config user.email "bot@github.com"
          git add logs/report.json || true
          git diff --staged --quiet || git commit -m "📊 Bot report $(date -u '+%Y-%m-%d %H:%M')"
          git push || true
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      - name: Send Notification
        if: always()
        env:
          TELEGRAM_TOKEN: ${{ secrets.TELEGRAM_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          SENDGRID_API_KEY: ${{ secrets.SENDGRID_API_KEY }}
          NOTIFY_EMAIL: ${{ secrets.NOTIFY_EMAIL }}
        run: python src/notify.py
        continue-on-error: true
