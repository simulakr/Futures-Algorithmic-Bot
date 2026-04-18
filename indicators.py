import numpy as np
import pandas as pd
from config import ATR_RANGES, Z_RANGES, Z_INDICATOR_PARAMS

# --- Temel Hesaplamalar ---

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()

def calculate_z(df: pd.DataFrame, symbol: str) -> pd.Series:
    pct_min, pct_max = Z_RANGES[symbol]
    mult = Z_INDICATOR_PARAMS["atr_multiplier"]
    return np.minimum(
        np.maximum(df["close"] * pct_min / 100, mult * df["atr"]),
        df["close"] * pct_max / 100,
    )

# --- Sadeleştirilmiş ZigZag ---

def calculate_atr_zigzag_2x(df: pd.DataFrame, atr_mult: float = 1.25) -> pd.DataFrame:
    closes, atrs = df["close"].values, df["z"].values
    n = len(df)
    
    high_pivot, low_pivot = [None] * n, [None] * n
    high_confirmed, low_confirmed = [0] * n, [0] * n
    
    last_price, last_pivot_idx, direction = closes[0], 0, None

    for i in range(1, n):
        price, atr = closes[i], atrs[i] * atr_mult
        if direction is None:
            if price >= last_price + atr: direction = "up"
            elif price <= last_price - atr: direction = "down"
        elif direction == "up":
            if price <= last_price - atr:
                high_pivot[last_pivot_idx], high_confirmed[i] = last_price, 1
                direction, last_price, last_pivot_idx = "down", price, i
            elif price > last_price: last_price, last_pivot_idx = price, i
        elif direction == "down":
            if price >= last_price + atr:
                low_pivot[last_pivot_idx], low_confirmed[i] = last_price, 1
                direction, last_price, last_pivot_idx = "up", price, i
            elif price < last_price: last_price, last_pivot_idx = price, i

    df["high_pivot_ff_2x"] = pd.Series(high_pivot).ffill()
    df["low_pivot_ff_2x"] = pd.Series(low_pivot).ffill()
    df["high_confirmed_2x"] = high_confirmed
    df["low_confirmed_2x"] = low_confirmed
    return df

# --- Market Yapısı (HH/HL/LH/LL) ---

def add_market_structure_2x(df: pd.DataFrame) -> pd.DataFrame:
    hff, lff = "high_pivot_ff_2x", "low_pivot_ff_2x"
    
    df["high_structure_2x"] = np.nan
    df.loc[df[hff] < df[hff].shift(1), "high_structure_2x"] = "LH"
    df.loc[df[hff] > df[hff].shift(1), "high_structure_2x"] = "HH"
    df["high_structure_2x"] = df["high_structure_2x"].ffill().fillna("HH")

    df["low_structure_2x"] = np.nan
    df.loc[df[lff] < df[lff].shift(1), "low_structure_2x"] = "LL"
    df.loc[df[lff] > df[lff].shift(1), "low_structure_2x"] = "HL"
    df["low_structure_2x"] = df["low_structure_2x"].ffill().fillna("LL")
    return df

# --- Ana Fonksiyon ---

def calculate_pivot_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    atr_low, atr_high = ATR_RANGES[symbol]
    
    # 1. Temel Göstergeler
    df["atr"] = calculate_atr(df)
    df["pct_atr"] = (df["atr"] / df["close"]) * 100
    df["z"] = calculate_z(df, symbol)
    
    # 2. ZigZag ve Yapı
    df = calculate_atr_zigzag_2x(df)
    df = add_market_structure_2x(df)
    
    # 3. Koşullar ve Maskeler
    atr_ok = (df["pct_atr"] > atr_low) & (df["pct_atr"] < atr_high)
    hff, lff = df["high_pivot_ff_2x"], df["low_pivot_ff_2x"]
    hcon, lcon = df["high_confirmed_2x"].astype(bool), df["low_confirmed_2x"].astype(bool)
    hs, ls = df["high_structure_2x"], df["low_structure_2x"]

    # 4. Sinyal Hesaplama (Pivot Breakout ve Go-Up Ayrımı)
    # Temel Breakoutlar (Yapı farketmeksizin fiyatın pivotu kırması)
    pivot_breakout_2x = lcon & hff.notna() & (df["close"] > hff) & atr_ok
    pivot_breakdown_2x = hcon & lff.notna() & (df["close"] < lff) & atr_ok

    # Go-Up Breakoutlar (Sadece güçlü HH+HL yapısı varken olan kırılımlar)
    pivot_goup_breakout_2x = pivot_breakout_2x & (ls == "HL") & (hs == "HH")
    pivot_goup_breakdown_2x = pivot_breakdown_2x & (hs == "LH") & (ls == "LL")

    # 5. İstenen Sonuç: NO-GOUP (Zayıf Yapılı Kırılımlar)
    df["pivot_no_goup_breakout_2x"] = pivot_breakout_2x & ~pivot_goup_breakout_2x
    df["pivot_no_goup_breakdown_2x"] = pivot_breakdown_2x & ~pivot_goup_breakdown_2x

    return df
