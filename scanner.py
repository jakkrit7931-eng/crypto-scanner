# scanner.py
# วางไฟล์นี้ไว้ที่ root ของ repo (ระดับเดียวกับ .github/)
# Token อ่านจาก Environment Variables (GitHub Secrets) — ไม่ต้อง hardcode

import os
import ccxt
import pandas as pd
import pandas_ta as ta
import requests
import time
import traceback
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG — อ่านจาก GitHub Secrets อัตโนมัติ
# ─────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

TIMEFRAMES  = ["4h", "1d"]
MAX_PAIRS   = 150
CANDLE_LIMIT = 250
MIN_SCORE   = 75

EMA_FAST  = 50
EMA_SLOW  = 200
RSI_LEN   = 14
RSI_OS    = 35
RSI_OB    = 65
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIG  = 9
BB_LEN    = 20
BB_MULT   = 2.0
SL_LOOKBACK = 10
RR_RATIO  = 3.0

# ─────────────────────────────────────────────
# EXCHANGE
# ─────────────────────────────────────────────
def create_exchange():
    return ccxt.binanceusdm({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

def get_usdt_pairs(exchange):
    markets = exchange.load_markets()
    pairs = [
        s for s, m in markets.items()
        if m.get("quote") == "USDT"
        and m.get("active", False)
        and m.get("type") == "swap"
        and not m.get("inverse", False)
    ]
    pairs.sort()
    return pairs[:MAX_PAIRS]

def fetch_ohlcv(exchange, symbol, timeframe):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=CANDLE_LIMIT)
        if len(ohlcv) < 210:
            return None
        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
    except Exception:
        return None

# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def calculate_indicators(df):
    df["ema50"]  = ta.ema(df["close"], length=EMA_FAST)
    df["ema200"] = ta.ema(df["close"], length=EMA_SLOW)
    df["rsi"]    = ta.rsi(df["close"], length=RSI_LEN)
    macd_df = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIG)
    df["macd_hist"] = macd_df[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIG}"]
    bb_df = ta.bbands(df["close"], length=BB_LEN, std=BB_MULT)
    df["bb_upper"] = bb_df[f"BBU_{BB_LEN}_{BB_MULT}"]
    df["bb_lower"] = bb_df[f"BBL_{BB_LEN}_{BB_MULT}"]
    return df.dropna()

# ─────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────
def analyze_signal(df):
    if len(df) < 3:
        return None
    c, c1, c2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]

    bull_trend  = c["close"] > c["ema50"]  and c["close"] > c["ema200"]
    bear_trend  = c["close"] < c["ema50"]  and c["close"] < c["ema200"]
    rsi_long    = (min(c1["rsi"], c2["rsi"]) < RSI_OS) and c["rsi"] > RSI_OS
    rsi_short   = (max(c1["rsi"], c2["rsi"]) > RSI_OB) and c["rsi"] < RSI_OB
    macd_bull   = c["macd_hist"] > 0  and c1["macd_hist"] <= 0
    macd_bear   = c["macd_hist"] < 0  and c1["macd_hist"] >= 0
    bb_long     = c1["low"]  <= c1["bb_lower"] and c["close"] > c["bb_lower"]
    bb_short    = c1["high"] >= c1["bb_upper"] and c["close"] < c["bb_upper"]

    long_score  = (25 if bull_trend else 0) + (25 if rsi_long  else 0) \
                + (25 if macd_bull  else 0) + (25 if bb_long   else 0)
    short_score = (25 if bear_trend else 0) + (25 if rsi_short else 0) \
                + (25 if macd_bear  else 0) + (25 if bb_short  else 0)

    if long_score >= short_score and long_score >= MIN_SCORE:
        direction, score = "LONG", long_score
        conditions = {"trend":bull_trend,"rsi":rsi_long,"macd":macd_bull,"bb":bb_long}
    elif short_score > long_score and short_score >= MIN_SCORE:
        direction, score = "SHORT", short_score
        conditions = {"trend":bear_trend,"rsi":rsi_short,"macd":macd_bear,"bb":bb_short}
    else:
        return None

    entry  = float(c["close"])
    recent = df.tail(SL_LOOKBACK)
    if direction == "LONG":
        sl   = float(recent["low"].min())
        risk = entry - sl
        tp1, tp2, tp3 = entry+risk, entry+risk*2, entry+risk*RR_RATIO
    else:
        sl   = float(recent["high"].max())
        risk = sl - entry
        tp1, tp2, tp3 = entry-risk, entry-risk*2, entry-risk*RR_RATIO

    if risk <= 0:
        return None

    return {"direction":direction,"score":score,"entry":entry,
            "sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "rsi":round(float(c["rsi"]),1),
            "macd_hist":round(float(c["macd_hist"]),6),
            "conditions":conditions}

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────
def fmt(price):
    if price >= 1000: return f"{price:,.2f}"
    if price >= 1:    return f"{price:.3f}"
    return f"{price:.6f}"

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram error: {e}")
        return False

def build_message(symbol, timeframe, signal):
    d     = signal["direction"]
    coin  = symbol.replace("/USDT:USDT","").replace("/USDT","")
    emoji = "🟢" if d == "LONG" else "🔴"
    arrow = "▲" if d == "LONG" else "▼"
    score = signal["score"]
    bar   = "█" * int(score/10) + "░" * (10 - int(score/10))
    cond  = signal["conditions"]
    ck    = lambda x: "✅" if x else "❌"

    return (
        f"{emoji} <b>{coin} — {d} {arrow}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"🕐 Timeframe : <b>{timeframe.upper()}</b>\n"
        f"📊 Score     : <b>{score}/100</b>  [{bar}]\n\n"
        f"<b>📌 Confluence</b>\n"
        f"  {ck(cond['trend'])} Trend  (EMA 50/200)\n"
        f"  {ck(cond['rsi'])}   RSI    ({signal['rsi']})\n"
        f"  {ck(cond['macd'])}  MACD   hist ({signal['macd_hist']})\n"
        f"  {ck(cond['bb'])}   BB     bounce\n\n"
        f"<b>💰 Trade Levels</b>\n"
        f"  Entry : <b>{fmt(signal['entry'])}</b>\n"
        f"  SL    : <code>{fmt(signal['sl'])}</code>  🛑\n"
        f"  TP1   : <code>{fmt(signal['tp1'])}</code>  🎯 (1R)\n"
        f"  TP2   : <code>{fmt(signal['tp2'])}</code>  🎯 (2R)\n"
        f"  TP3   : <code>{fmt(signal['tp3'])}</code>  🎯 (1:{int(RR_RATIO)}R)\n\n"
        f"⚠️ <i>ไม่ใช่คำแนะนำทางการเงิน — DYOR</i>\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )

# ─────────────────────────────────────────────
# MAIN — รันครั้งเดียวแล้วจบ (GitHub Actions style)
# ─────────────────────────────────────────────
def main():
    print(f"🔍 SCAN START {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    exchange = create_exchange()
    pairs    = get_usdt_pairs(exchange)
    print(f"✅ {len(pairs)} pairs")

    alerts = 0
    for idx, symbol in enumerate(pairs, 1):
        coin = symbol.replace("/USDT:USDT","").replace("/USDT","")
        print(f"[{idx:>3}/{len(pairs)}] {coin:<12}", end="")
        for tf in TIMEFRAMES:
            try:
                df = fetch_ohlcv(exchange, symbol, tf)
                if df is None:
                    print(f" [{tf}:skip]", end=""); continue
                df     = calculate_indicators(df)
                signal = analyze_signal(df)
                if signal is None:
                    print(f" [{tf}:-]", end=""); continue
                print(f" [{tf}:{signal['direction']} {signal['score']}]", end="")
                msg = build_message(symbol, tf, signal)
                if send_telegram(msg):
                    alerts += 1
                time.sleep(0.5)
            except Exception:
                print(f" [{tf}:ERR]", end="")
            time.sleep(0.1)
        print()

    summary = (
        f"📋 <b>Scan สรุป</b>\n"
        f"━━━━━━━━━━━━━━━\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"📊 pairs : {len(pairs)}\n"
        f"📨 alerts: {alerts}"
    )
    send_telegram(summary)
    print(f"\n✅ DONE — alerts: {alerts}")

if __name__ == "__main__":
    main()
