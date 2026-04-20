import numpy as np
import pandas as pd
import warnings

from config import ATR_RANGES, Z_RANGES, Z_INDICATOR_PARAMS

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── Temel Göstergeler ────────────────────────────────────────────────────────

def calculate_rsi(df: pd.DataFrame, window: int = 14, col: str = "close") -> pd.Series:
    delta    = df[col].diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    prev  = close.shift(1)
    tr    = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def calculate_sma(df: pd.DataFrame, window: int, col: str = "close") -> pd.Series:
    return df[col].rolling(window).mean()


def calculate_donchian(df: pd.DataFrame, window: int) -> pd.DataFrame:
    upper  = df["high"].rolling(window).max()
    lower  = df["low"].rolling(window).min()
    middle = (upper + lower) / 2
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": middle})


def calculate_nadaraya_watson(
    df:         pd.DataFrame,
    bandwidth:  float = 8.0,
    multiplier: float = 3.0,
    col:        str   = "close",
    window:     int   = 50,
) -> pd.DataFrame:
    n      = len(df)
    src    = df[col].values

    def gauss(x, h):
        return np.exp(-(x ** 2) / (h * h * 2))

    weights     = np.array([gauss(i, bandwidth) for i in range(window)])
    weights_sum = weights.sum()

    nw_mid   = np.full(n, np.nan)
    nw_upper = np.full(n, np.nan)
    nw_lower = np.full(n, np.nan)

    for i in range(window - 1, n):
        wsum      = np.dot(src[i - window + 1 : i + 1], weights[::-1])
        mid       = wsum / weights_sum
        nw_mid[i] = mid
        mae       = np.mean(np.abs(src[i - window + 1 : i + 1] - nw_mid[i - window + 1 : i + 1])) * multiplier
        nw_upper[i] = mid + mae
        nw_lower[i] = mid - mae

    return pd.DataFrame({"nw": nw_mid, "nw_upper": nw_upper, "nw_lower": nw_lower}, index=df.index)


# ─── Z Göstergesi ─────────────────────────────────────────────────────────────

def calculate_z(df: pd.DataFrame, symbol: str) -> pd.Series:
    if symbol not in Z_RANGES:
        raise ValueError(f"Z_RANGES içinde {symbol} tanımlı değil.")
    pct_min, pct_max = Z_RANGES[symbol]
    mult = Z_INDICATOR_PARAMS["atr_multiplier"]
    return np.minimum(
        np.maximum(df["close"] * pct_min / 100, mult * df["atr"]),
        df["close"] * pct_max / 100,
    )


# ─── ATR ZigZag ───────────────────────────────────────────────────────────────

def calculate_atr_zigzag(df: pd.DataFrame, atr_col: str = "atr", atr_mult: float = 1.0, suffix: str = "") -> pd.DataFrame:
    closes = df["close"].values
    atrs   = df[atr_col].values
    n      = len(df)

    high_pivot     = [None] * n
    low_pivot      = [None] * n
    high_pivot_atr = [None] * n
    low_pivot_atr  = [None] * n
    high_confirmed = [0] * n
    low_confirmed  = [0] * n
    pivot_bars_ago = [None] * n

    last_price     = closes[0]
    last_pivot_idx = 0
    direction      = None

    for i in range(1, n):
        price = closes[i]
        atr   = atrs[i] * atr_mult

        if direction is None:
            if price >= last_price + atr:
                direction = "up"
                high_pivot[last_pivot_idx]     = closes[last_pivot_idx]
                high_pivot_atr[last_pivot_idx] = atrs[last_pivot_idx]
            elif price <= last_price - atr:
                direction = "down"
                low_pivot[last_pivot_idx]     = closes[last_pivot_idx]
                low_pivot_atr[last_pivot_idx] = atrs[last_pivot_idx]

        elif direction == "up":
            if price <= last_price - atr:
                high_pivot[last_pivot_idx]     = last_price
                high_pivot_atr[last_pivot_idx] = atrs[last_pivot_idx]
                high_confirmed[i] = 1
                pivot_bars_ago[i] = i - last_pivot_idx
                direction      = "down"
                last_price     = price
                last_pivot_idx = i
            elif price > last_price:
                last_price     = price
                last_pivot_idx = i

        elif direction == "down":
            if price >= last_price + atr:
                low_pivot[last_pivot_idx]     = last_price
                low_pivot_atr[last_pivot_idx] = atrs[last_pivot_idx]
                low_confirmed[i] = 1
                pivot_bars_ago[i] = i - last_pivot_idx
                direction      = "up"
                last_price     = price
                last_pivot_idx = i
            elif price < last_price:
                last_price     = price
                last_pivot_idx = i

    s = suffix
    df[f"high_pivot{s}"]     = high_pivot
    df[f"low_pivot{s}"]      = low_pivot
    df[f"high_pivot_atr{s}"] = high_pivot_atr
    df[f"low_pivot_atr{s}"]  = low_pivot_atr
    df[f"high_confirmed{s}"] = high_confirmed
    df[f"low_confirmed{s}"]  = low_confirmed
    df[f"pivot_bars_ago{s}"] = pivot_bars_ago

    # Forward fill
    df[f"high_pivot_ff{s}"]     = df[f"high_pivot{s}"].ffill()
    df[f"low_pivot_ff{s}"]      = df[f"low_pivot{s}"].ffill()
    df[f"high_pivot_atr_ff{s}"] = df[f"high_pivot_atr{s}"].ffill()
    df[f"low_pivot_atr_ff{s}"]  = df[f"low_pivot_atr{s}"].ffill()

    for base, col in [("high_confirmed", f"high_confirmed{s}"), ("low_confirmed", f"low_confirmed{s}")]:
        tmp = df[col].replace(0, np.nan).ffill()
        df[f"{col}_ff"] = tmp.fillna(0).astype(int)

    # Pivot bars ago forward fill (sürekli artan)
    filled = []
    last_val = last_idx = None
    for i, v in enumerate(pivot_bars_ago):
        if v is not None:
            last_val, last_idx = v, i
            filled.append(v)
        elif last_val is not None:
            filled.append(last_val + (i - last_idx))
        else:
            filled.append(None)
    df[f"pivot_bars_ago_ff{s}"] = filled

    return df


# ─── Market Yapısı (HH/HL/LH/LL) ─────────────────────────────────────────────

def add_market_structure(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    s = suffix
    high_ff = f"high_pivot_ff{s}"
    low_ff  = f"low_pivot_ff{s}"

    df[f"high_structure{s}"] = np.nan
    df.loc[df[high_ff] < df[high_ff].shift(1), f"high_structure{s}"] = "LH"
    df.loc[df[high_ff] > df[high_ff].shift(1), f"high_structure{s}"] = "HH"
    df[f"high_structure{s}"] = df[f"high_structure{s}"].ffill().fillna("HH")

    df[f"low_structure{s}"] = np.nan
    df.loc[df[low_ff] < df[low_ff].shift(1), f"low_structure{s}"] = "LL"
    df.loc[df[low_ff] > df[low_ff].shift(1), f"low_structure{s}"] = "HL"
    df[f"low_structure{s}"] = df[f"low_structure{s}"].ffill().fillna("LL")

    return df


# ─── Shift Koşulu Yardımcısı ──────────────────────────────────────────────────

def _build_shift_ok(df: pd.DataFrame, price_col: str, pivot_col: str, direction: str, n: int = 5) -> pd.Series:
    """
    Son n barda fiyatın pivot seviyesinin belirtilen tarafında kaldığını kontrol eder.
    direction="long"  -> close.shift(i) < pivot_col (önceki barlar pivot altında)
    direction="short" -> close.shift(i) > pivot_col (önceki barlar pivot üstünde)
    """
    conditions = []
    for i in range(1, n + 1):
        if direction == "long":
            cond = (df[pivot_col].notna() & (df[price_col].shift(i) < df[pivot_col])).fillna(False)
        else:
            cond = (df[pivot_col].notna() & (df[price_col].shift(i) > df[pivot_col])).fillna(False)
        conditions.append(cond)
    return pd.concat(conditions, axis=1).all(axis=1)


# ─── Ana Hesaplama ────────────────────────────────────────────────────────────

def calculate_indicators(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    atr_low, atr_high = ATR_RANGES[symbol]

    # Temel göstergeler
    df["rsi"]     = calculate_rsi(df)
    df["atr"]     = calculate_atr(df)
    df["pct_atr"] = (df["atr"] / df["close"]) * 100

    df["z"]     = calculate_z(df, symbol)
    df["pct_z"] = (df["z"] / df["close"]) * 100

    # Donchian Kanalları
    for w in [20, 50]:
        dc = calculate_donchian(df, w)
        df[f"dc_upper_{w}"]          = dc["upper"]
        df[f"dc_lower_{w}"]          = dc["lower"]
        df[f"dc_middle_{w}"]         = dc["middle"]
        df[f"dc_position_ratio_{w}"] = (df["close"] - dc["lower"]) / (dc["upper"] - dc["lower"]) * 100
        df[f"dc_breakout_{w}"]       = df["high"] > dc["upper"]
        df[f"dc_breakdown_{w}"]      = df["low"]  < dc["lower"]

    # SMA & Trend
    df["sma_50"]       = calculate_sma(df, 50)
    df["sma_200"]      = calculate_sma(df, 200)
    df["trend_50_200"] = np.where(df["sma_50"] > df["sma_200"], "uptrend", "downtrend")

    # Nadaraya-Watson Envelope
    nw = calculate_nadaraya_watson(df)
    df[["nw", "nw_upper", "nw_lower"]] = nw

    # ATR ZigZag (_2x ve _3x)
    df = calculate_atr_zigzag(df, atr_col="z", atr_mult=1.25, suffix="_2x")
    df = calculate_atr_zigzag(df, atr_col="z", atr_mult=3.0,  suffix="_3x")

    # Market Yapısı
    df = add_market_structure(df, "_2x")
    df = add_market_structure(df, "_3x")

    # ATR filtre maskesi
    atr_ok = (atr_low < df["pct_atr"]) & (df["pct_atr"] < atr_high)

    # ─── Ortak shift koşulları (_2x) ─────────────────────────────────────────

    long_shift_ok  = _build_shift_ok(df, "close", "high_pivot_ff_2x", "long",  n=5)
    short_shift_ok = _build_shift_ok(df, "close", "low_pivot_ff_2x",  "short", n=5)

    # ─── Pivot Go Sinyalleri ──────────────────────────────────────────────────

    df["pivot_go_up_2x"]   = False
    df["pivot_go_down_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        (df["low_structure_2x"]  == "HL") &
        (df["high_structure_2x"] == "HH") &
        (df["trend_50_200"]      == "uptrend") &
        (df["close"] < df["nw_upper"]) & atr_ok,
        "pivot_go_up_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        (df["high_structure_2x"] == "LH") &
        (df["low_structure_2x"]  == "LL") &
        (df["trend_50_200"]      == "downtrend") &
        (df["close"] > df["nw_lower"]) & atr_ok,
        "pivot_go_down_2x",
    ] = True

    # _3x: sadece market structure, sinyal yok
    df["pivot_go_up_3x"]   = False
    df["pivot_go_down_3x"] = False

    df.loc[
        df["low_confirmed_3x"].astype(bool) &
        (df["low_structure_3x"]  == "HL") &
        (df["high_structure_3x"] == "HH") &
        (df["close"] < df["nw_upper"]) & atr_ok,
        "pivot_go_up_3x",
    ] = True

    df.loc[
        df["high_confirmed_3x"].astype(bool) &
        (df["high_structure_3x"] == "LH") &
        (df["low_structure_3x"]  == "LL") &
        (df["close"] > df["nw_lower"]) & atr_ok,
        "pivot_go_down_3x",
    ] = True

    # ─── pivot_go_breakout (_2x) ──────────────────────────────────────────────
    # Yapı: HL low + high_structure != HH + pivot geçişi

    df["pivot_go_breakout_2x"]  = False
    df["pivot_go_breakdown_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        (df["low_structure_2x"]  == "HL") &
        (df["high_structure_2x"] != "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) & atr_ok,
        "pivot_go_breakout_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        (df["high_structure_2x"] == "LH") &
        (df["low_structure_2x"]  != "LL") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) & atr_ok,
        "pivot_go_breakdown_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "HL") &
        long_shift_ok &
        (df["high_structure_2x"] != "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_go_breakout_2x"]),
        "pivot_go_breakout_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  != "LL") &
        short_shift_ok &
        (df["high_structure_2x"] == "LH") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_go_breakdown_2x"]),
        "pivot_go_breakdown_2x",
    ] = True

    # _3x
    df["pivot_go_breakout_3x"]  = False
    df["pivot_go_breakdown_3x"] = False

    df.loc[
        df["low_confirmed_3x"].astype(bool) &
        (df["low_structure_3x"]  == "HL") &
        (df["high_structure_3x"] != "HH") &
        df["high_pivot_ff_3x"].notna() &
        (df["close"] > df["high_pivot_ff_3x"]) & atr_ok,
        "pivot_go_breakout_3x",
    ] = True

    df.loc[
        df["high_confirmed_3x"].astype(bool) &
        (df["high_structure_3x"] == "LH") &
        (df["low_structure_3x"]  != "LL") &
        df["low_pivot_ff_3x"].notna() &
        (df["close"] < df["low_pivot_ff_3x"]) & atr_ok,
        "pivot_go_breakdown_3x",
    ] = True

    # ─── pivot_goup_breakout (_2x) ────────────────────────────────────────────
    # Yapı: HL low + HH high + pivot geçişi (yapı teyitli)

    df["pivot_goup_breakout_2x"]  = False
    df["pivot_goup_breakdown_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        (df["low_structure_2x"]  == "HL") &
        (df["high_structure_2x"] == "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) & atr_ok,
        "pivot_goup_breakout_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        (df["high_structure_2x"] == "LH") &
        (df["low_structure_2x"]  == "LL") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) & atr_ok,
        "pivot_goup_breakdown_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "HL") &
        long_shift_ok &
        (df["high_structure_2x"] == "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_goup_breakout_2x"]),
        "pivot_goup_breakout_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "LL") &
        short_shift_ok &
        (df["high_structure_2x"] == "LH") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_goup_breakdown_2x"]),
        "pivot_goup_breakdown_2x",
    ] = True

    # ─── pivot_choch_breakout (_2x) ───────────────────────────────────────────
    # Yapı: LL low + LH high + pivot geçişi (change of character: zayıf downtrend kırılıyor)

    df["pivot_choch_breakout_2x"]  = False
    df["pivot_choch_breakdown_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        (df["low_structure_2x"]  == "LL") &
        (df["high_structure_2x"] == "LH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) & atr_ok,
        "pivot_choch_breakout_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        (df["high_structure_2x"] == "HH") &
        (df["low_structure_2x"]  == "HL") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) & atr_ok,
        "pivot_choch_breakdown_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "LL") &
        long_shift_ok &
        (df["high_structure_2x"] == "LH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_choch_breakout_2x"]),
        "pivot_choch_breakout_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "HL") &
        short_shift_ok &
        (df["high_structure_2x"] == "HH") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_choch_breakdown_2x"]),
        "pivot_choch_breakdown_2x",
    ] = True

    # ─── pivot_hhll_breakout (_2x) ────────────────────────────────────────────
    # Yapı: LL low + HH high + pivot geçişi (mixed yapı: güçlü high ama kötüleşen low)

    df["pivot_hhll_breakout_2x"]  = False
    df["pivot_hhll_breakdown_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        (df["low_structure_2x"]  == "LL") &
        (df["high_structure_2x"] == "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) & atr_ok,
        "pivot_hhll_breakout_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        (df["high_structure_2x"] == "HH") &
        (df["low_structure_2x"]  == "LL") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) & atr_ok,
        "pivot_hhll_breakdown_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "LL") &
        long_shift_ok &
        (df["high_structure_2x"] == "HH") &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_hhll_breakout_2x"]),
        "pivot_hhll_breakout_2x",
    ] = True

    df.loc[
        (df["low_structure_2x"]  == "LL") &
        short_shift_ok &
        (df["high_structure_2x"] == "HH") &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_hhll_breakdown_2x"]),
        "pivot_hhll_breakdown_2x",
    ] = True

    # ─── pivot_breakout (_2x) ─────────────────────────────────────────────────
    # Market structure koşulsuz: sadece pivot geçişi + atr_ok

    df["pivot_breakout_2x"]  = False
    df["pivot_breakdown_2x"] = False

    df.loc[
        df["low_confirmed_2x"].astype(bool) &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) & atr_ok,
        "pivot_breakout_2x",
    ] = True

    df.loc[
        df["high_confirmed_2x"].astype(bool) &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) & atr_ok,
        "pivot_breakdown_2x",
    ] = True

    df.loc[
        long_shift_ok &
        df["high_pivot_ff_2x"].notna() &
        (df["close"] > df["high_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_breakout_2x"]),
        "pivot_breakout_2x",
    ] = True

    df.loc[
        short_shift_ok &
        df["low_pivot_ff_2x"].notna() &
        (df["close"] < df["low_pivot_ff_2x"]) &
        atr_ok & (~df["pivot_breakdown_2x"]),
        "pivot_breakdown_2x",
    ] = True

    # ─── pivot_no_goup (_2x) ──────────────────────────────────────────────────
    # pivot_breakout var ama pivot_goup_breakout yok (yapı teyitsiz geçiş)

    df["pivot_no_goup_breakout_2x"]  = (df["pivot_breakout_2x"]  == True) & (df["pivot_goup_breakout_2x"]  == False)
    df["pivot_no_goup_breakdown_2x"] = (df["pivot_breakdown_2x"] == True) & (df["pivot_goup_breakdown_2x"] == False)

    return df
