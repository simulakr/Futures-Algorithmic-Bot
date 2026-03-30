# 6-3 Go Breakout-Breakdown
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Binance API ──────────────────────────────────────────────────────────────
BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TESTNET            = os.getenv("TESTNET", "false").lower() == "true"

# ─── Semboller & Zaman Aralığı ────────────────────────────────────────────────
SYMBOLS  = [ "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"] #"BNBUSDT"
INTERVAL = "15m"

# ─── Percent ATR Filtreleri ───────────────────────────────────────────────────
ATR_RANGES = {
    "BTCUSDT":  (0.173, 0.645),
    "ETHUSDT":  (0.363, 0.990),
    "SOLUSDT":  (0.423, 1.176),
    "XRPUSDT":  (0.363, 1.378),
    "DOGEUSDT": (0.465, 1.306),
}

# ─── Z Göstergesi: atr.quantile(0.35 - 0.65) ────────────────────────────────────────────────────────────
Z_RANGES = {
    'BTCUSDT': (0.2109, 0.309),
    'ETHUSDT': (0.372, 0.523),
    'SOLUSDT': (0.451, 0.622),
    'DOGEUSDT': (0.498, 0.698),
    'XRPUSDT': (0.397, 0.604),
}

Z_INDICATOR_PARAMS = {
    "atr_period":     14,
    "atr_multiplier": 1,
}

# ─── Pozisyon Büyüklüğü ───────────────────────────────────────────────────────
QUANTITY_PRECISION = {
    "BTCUSDT":  3,
    "ETHUSDT":  2,
    "SOLUSDT":  1,
    "XRPUSDT":  0,
    "DOGEUSDT": 0,
}

PRICE_PRECISION = {
    "BTCUSDT":  2,
    "ETHUSDT":  2,
    "SOLUSDT":  3,
    "XRPUSDT":  4,
    "DOGEUSDT": 5,
}


# ─── Risk & Kaldıraç ─────────────────────────────────────────────────────────

DEFAULT_LEVERAGE  = 20
DEFAULT_RISK_USDT = 10

SYMBOL_SETTINGS = {
    "BTCUSDT":  {"risk": 10, "leverage": 20},
    "ETHUSDT":  {"risk": 10, "leverage": 20},
    "SOLUSDT":  {"risk": 10, "leverage": 20},
    "XRPUSDT":  {"risk": 10, "leverage": 20},
    "DOGEUSDT": {"risk": 10, "leverage": 20},
}

# ─── TP / SL Çarpanları ───────────────────────────────────────────────────────
TP_ATR_MULTIPLIER = 6
SL_ATR_MULTIPLIER = 3
