import os
import pandas as pd
import numpy as np


def calc_weighted_score(close: np.ndarray) -> np.ndarray:
    """
    對單一股票的 close 價格序列，計算每天的 weightedScore
    使用：
    3m = 63 天
    6m = 126 天
    9m = 189 天
    12m = 252 天
    """
    n = close.shape[0]
    weighted = np.full(n, np.nan, dtype=np.float32)
    if n <= 252:
        return weighted

    tail = close[252:]
    with np.errstate(divide="ignore", invalid="ignore"):
        length = n - 252
        c = close  # alias
        ret_3m = c[252:] / c[189:189+length] - 1.0
        ret_6m = c[252:] / c[126:126+length] - 1.0
        ret_9m = c[252:] / c[63:63+length] - 1.0
        ret_12m = c[252:] / c[0:length] - 1.0
        weighted[252:] = (ret_3m * 0.4 + ret_6m * 0.2 +
                          ret_9m * 0.2 + ret_12m * 0.2).astype(np.float32)

    return weighted


def calculate_delta_rs(rs, window: int = 10) -> np.ndarray:
    """
    用 rolling 線性回歸斜率計算 ΔRS（單一股票的 rs_rating 序列）

    Parameters
    ----------
    rs     : 單一股票的 rsRating 時間序列（已按日期排序）
    window : 回歸窗口，預設 10 個交易日

    Returns
    -------
    slopes : 與 rs 等長的 np.ndarray，前 window-1 筆為 NaN
    """
    n = len(rs)
    slopes = np.full(n, np.nan, dtype=np.float32)

    # 預先建好 X，避免迴圈內重複建立
    X = np.arange(window, dtype=np.float32)
    X_mean = X.mean()
    X_var = ((X - X_mean) ** 2).sum()   # Σ(x - x̄)²

    for i in range(window - 1, n):
        y = rs[i - window + 1: i + 1]
        if np.isnan(y).any():
            continue
        # OLS 斜率 = Σ(x-x̄)(y-ȳ) / Σ(x-x̄)²
        slopes[i] = ((X - X_mean) * (y - y.mean())).sum() / X_var

    return slopes


def calc_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    計算 ATR（Average True Range），使用 Wilder's smoothing（EMA with alpha=1/window）。

    True Range = max(
        high - low,
        |high - prev_close|,
        |low  - prev_close|
    )

    Parameters
    ----------
    high, low, close : 等長的價格序列（已按日期排序）
    window           : ATR 平滑週期，預設 14

    Returns
    -------
    atr : 與輸入等長的 np.ndarray（float32），前 window 筆為 NaN
    """
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float32)
    if n < window + 1:
        return atr

    # ── True Range ──────────────────────────────────────────
    prev_close = close[:-1]          # shape (n-1,)
    hl = high[1:] - low[1:]        # high - low
    hpc = np.abs(high[1:] - prev_close)   # |high - prev_close|
    lpc = np.abs(low[1:] - prev_close)   # |low  - prev_close|

    tr = np.maximum(hl, np.maximum(hpc, lpc)).astype(np.float32)
    # tr 的索引對應原序列的 [1..n-1]

    # ── Wilder's smoothing（等同 EMA alpha=1/window）────────
    # 第一個 ATR 值 = 前 window 筆 TR 的簡單平均
    first_atr = tr[:window].mean()
    alpha = 1.0 / window

    wilder = np.empty(len(tr), dtype=np.float32)
    wilder[window - 1] = first_atr

    # 向量化 Wilder smoothing（用迴圈仍是最直覺，但加 numba 可加速）
    for i in range(window, len(tr)):
        wilder[i] = wilder[i - 1] * (1.0 - alpha) + tr[i] * alpha

    # 對齊回原序列（tr 從索引 1 開始，ATR 從索引 window 開始）
    atr[window:] = wilder[window - 1:]

    return atr


def build_features(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict[str, np.ndarray]:
    close_series = pd.Series(close, copy=False)

    atr_raw = calc_atr(high, low, close, window=14)  # 原始 ATR（同幣值單位）
    # ATR% = ATR / close，方便跨股票比較（去除價格量級差異）
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = (atr_raw / close.astype(np.float32)).astype(np.float32)
    return {
        "roc5": close_series.pct_change(5).to_numpy(dtype=np.float32),
        "roc20": close_series.pct_change(20).to_numpy(dtype=np.float32),
        "ma5": close_series.rolling(5).mean().to_numpy(dtype=np.float32),
        "ma20": close_series.rolling(20).mean().to_numpy(dtype=np.float32),
        "volatility": close_series.pct_change().rolling(20).std().to_numpy(dtype=np.float32),
        "atr":       atr_raw,    # 原始 ATR（同幣值單位）
        "atr_pct":   atr_pct,    # ATR / close（百分比波動，跨股票可比）
    }


def label_sample(df: pd.DataFrame, k: float = 1.0, max_hold_days: int = 63) -> pd.DataFrame:
    """
    對 df 的每一行（每個交易日），以「隔天開盤」作為進場價，
    往後最多 max_hold_days 天，標記 1 / 0 / -1。

    Parameters
    ----------
    df            : 含 open, close, atr 欄位，已按日期升序排列的單支股票 DataFrame
    k             : 風險係數，R = k * ATR
    max_hold_days : 最大持有天數（預設 63 ≈ 3 個月）

    Returns
    -------
    df 加上 'label' 欄位（int8），最後 max_hold_days+1 筆因無足夠未來資料設為 NaN
    """
    n = len(df)
    open_arr = df["open"].to_numpy(dtype=np.float32)
    close_arr = df["close"].to_numpy(dtype=np.float32)
    high_arr = df["max"].to_numpy(dtype=np.float32)
    low_arr = df["min"].to_numpy(dtype=np.float32)
    atr_arr = df["atr"].to_numpy(dtype=np.float32)

    labels = np.full(n, np.nan, dtype=object)  # 預設 NaN（不足未來資料的行）

    # 最後 (max_hold_days + 1) 筆無法完整往前看，跳過
    # i     = 訊號日（當天收盤後決定是否進場）
    # i+1   = 進場日（隔天開盤進場）
    # i+1 ~ i+max_hold_days = 持有期間
    for i in range(n - max_hold_days - 1):
        entry_price = open_arr[i + 1]
        atr_val = atr_arr[i + 1]

        # ATR 或進場價為 NaN 時跳過
        if np.isnan(entry_price) or np.isnan(atr_val) or atr_val <= 0:
            continue

        R = k * atr_val
        target = entry_price + 2.0 * R
        stop_loss = entry_price - R

        label = 0  # 預設：超過持有期，視為中性

        for j in range(1, max_hold_days + 1):
            h = high_arr[i + 1 + j]
            l = low_arr[i + 1 + j]
            if np.isnan(h) or np.isnan(l):
                continue
            if l <= stop_loss:
                label = -1
                break
            elif h >= target:
                label = 1
                break

        labels[i] = label

    df = df.copy()
    df["label"] = pd.array(labels, dtype=pd.Int8Dtype())  # 支援 NaN 的整數型別
    return df


def dataProcess(dataTAIEX):
    """計算每天所有股票的 RS Rating，並存成 cache/{today}_Data.pkl"""
    all_stock_scores = []

    # 用 scandir 比 listdir 更有效率
    for entry in os.scandir("data"):
        if not entry.is_file():
            continue

        file_name = entry.name
        if len(file_name) != 30:
            continue

        df = pd.read_pickle(entry.path)
        df = df[df["date"] >= "2024-01-01"]  # 只處理 2024 年以後的資料
        if len(df) <= 252:
            continue

        # 只在真的沒排序時才排序
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        close_np = df["close"].to_numpy(dtype=np.float64, copy=False)
        high_np = df["max"].to_numpy(dtype=np.float64, copy=False)
        low_np = df["min"].to_numpy(dtype=np.float64, copy=False)

        weighted_score = calc_weighted_score(close_np)
        valid_mask = ~np.isnan(weighted_score)
        if not valid_mask.any():
            continue

        features = build_features(high_np, low_np, close_np)

        date_arr = df["date"].to_numpy()
        # entry_date = np.empty(date_arr.shape[0], dtype=object)
        # entry_date[:-1] = date_arr[1:]
        # entry_date[-1] = pd.NaT

        # open_arr = df["open"].to_numpy(dtype=np.float32, copy=False)
        # entry_price = np.empty(open_arr.shape[0], dtype=np.float32)
        # entry_price[:-1] = open_arr[1:]
        # entry_price[-1] = np.nan

        temp = pd.DataFrame({
            "stock_id": df["stock_id"].iloc[0],
            "date": date_arr[valid_mask],
            "volume": df["Trading_Volume"].to_numpy(dtype=np.int32, copy=False)[valid_mask],
            "volatility": features["volatility"][valid_mask],
            "weightedScore": weighted_score[valid_mask],
            "close": close_np.astype(np.float32, copy=False)[valid_mask],
            "open": df["open"].to_numpy(dtype=np.float32, copy=False)[valid_mask],
            "max": high_np.astype(np.float32, copy=False)[valid_mask],
            "min": low_np.astype(np.float32, copy=False)[valid_mask],
            # "entryDate": entry_date[valid_mask],
            # "entryPrice": entry_price[valid_mask],
            "roc5": features["roc5"][valid_mask],
            "roc20": features["roc20"][valid_mask],
            "ma5": features["ma5"][valid_mask],
            "ma20": features["ma20"][valid_mask],
            "atr":          features["atr"][valid_mask],
            "atr_pct":      features["atr_pct"][valid_mask]
        })
        temp = temp.sort_values("date").reset_index(drop=True)  # 確保日期升序
        temp = label_sample(temp, k=1.0, max_hold_days=21)
        all_stock_scores.append(temp)

    print(f"已處理 {len(all_stock_scores)} 支股票的 weightedScore 計算")
    if not all_stock_scores:
        print("沒有可用資料")
        return pd.DataFrame()

    # 一次合併
    big_df = pd.concat(all_stock_scores, ignore_index=True)
    big_df = big_df.merge(dataTAIEX, on="date", how="left")  # 把台股大盤資料合併進來
    # ── 同一天橫截面排名 ──────────────────────────────────────
    print("正在計算同一天的 RS Rating 百分位排名...")
    big_df["rsRating"] = (
        big_df.groupby("date", sort=False)["weightedScore"]
        .rank(pct=True, method="min")
        .mul(100)
        .astype(np.uint8)
    )

    # ── ΔRS：每支股票自己的時間序列斜率 ──────────────────────
    # 先排好時間順序再 groupby，確保斜率計算正確
    print("正在計算 ΔRS...")
    big_df = big_df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    big_df["deltaRS"] = (
        big_df.groupby("stock_id", sort=False)["rsRating"]
        .transform(lambda x: calculate_delta_rs(x.to_numpy(dtype=np.float32)))
    )

    # ── ΔRS Rank：同日橫截面百分位（0~100）────────────────────
    big_df["deltaRS_rank"] = (
        big_df.groupby("date", sort=False)["deltaRS"]
        .transform(lambda x: x.rank(pct=True, method="min") * 100)
        .astype(np.float32)
    )

    # ── 最終排序 ──────────────────────────────────────────────
    big_df = big_df.sort_values(
        ["date", "rsRating"],
        ascending=[True, False],
        kind="mergesort"
    ).reset_index(drop=True)

    big_df.drop(columns=["weightedScore"], inplace=True)  # 不再需要原始分數

    # 存檔
    os.makedirs("cache", exist_ok=True)
    today = big_df["date"].max()
    big_df.to_pickle(f"cache/{today}_Data.pkl")
    print(big_df["label"].value_counts(dropna=False))
    return big_df


def dataProcessTAIEX():
    dfPath = None
    for entry in os.scandir("data"):
        if not entry.is_file():
            continue

        file_name = entry.name
        if file_name.endswith("TAIEX.pkl"):
            dfPath = entry.path
            break
    if dfPath is None:
        print("找不到 TAIEX 的資料檔")
        return None
    df = pd.read_pickle(dfPath)
    close_np = df["close"].to_numpy(dtype=np.float64, copy=False)
    high_np = df["max"].to_numpy(dtype=np.float64, copy=False)
    low_np = df["min"].to_numpy(dtype=np.float64, copy=False)
    features = build_features(high_np, low_np, close_np)

    temp = pd.DataFrame({
        "date": df["date"],
        "TAIEXvolume": df["Trading_Volume"].to_numpy(dtype=np.int32, copy=False),
        "TAIEXvolatility": features["volatility"],
        "TAIEXclose": close_np.astype(np.float32, copy=False),
        "TAIEXroc5": features["roc5"],
        "TAIEXroc20": features["roc20"],
        "TAIEXma5": features["ma5"],
        "TAIEXma20": features["ma20"],
        "TAIEXtrend": features["roc5"] > features["roc20"],
        "TAIEXatr": features["atr"],
        "TAIEXatr_pct": features["atr_pct"]
    })
    return temp


if __name__ == "__main__":
    dataProcess(dataProcessTAIEX())
