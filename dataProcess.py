import os
import pandas as pd
import numpy as np
import numba


def calcWeightedScore(close: np.ndarray) -> np.ndarray:
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

    with np.errstate(divide="ignore", invalid="ignore"):
        length = n - 252
        c = close

        ret3m = c[252:] / c[189:189 + length] - 1.0
        ret6m = c[252:] / c[126:126 + length] - 1.0
        ret9m = c[252:] / c[63:63 + length] - 1.0
        ret12m = c[252:] / c[0:length] - 1.0

        weighted[252:] = (
            ret3m * 0.4
            + ret6m * 0.2
            + ret9m * 0.2
            + ret12m * 0.2
        ).astype(np.float32)

    return weighted


def calculateDeltaRs(rs, window: int = 10) -> np.ndarray:
    """
    向量化滾動 OLS 斜率。

    用 rolling 線性回歸斜率計算 ΔRS。

    Parameters
    ----------
    rs:
        單一股票的 rsRating 時間序列，已按日期排序。

    window:
        回歸窗口，預設 10 個交易日。

    Returns
    -------
    slopes:
        與 rs 等長的 np.ndarray，前 window - 1 筆為 NaN。
    """
    s = pd.Series(rs, dtype=np.float32)

    x = np.arange(window, dtype=np.float32)
    xMean = x.mean()
    xVar = ((x - xMean) ** 2).sum()

    yRollSum = s.rolling(window, min_periods=window).sum()

    weights = np.arange(window, dtype=np.float32)
    xyRoll = s.rolling(window, min_periods=window).apply(
        lambda y: (weights * y).sum(),
        raw=True
    )

    slopes = (xyRoll - xMean * yRollSum) / xVar

    return slopes.to_numpy(dtype=np.float32)


def calcAtr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14
) -> np.ndarray:
    """
    計算 ATR，使用 Wilder's smoothing。

    True Range = max(
        high - low,
        |high - prevClose|,
        |low - prevClose|
    )

    Parameters
    ----------
    high, low, close:
        等長的價格序列，已按日期排序。

    window:
        ATR 平滑週期，預設 14。

    Returns
    -------
    atr:
        與輸入等長的 np.ndarray，float32，前 window 筆為 NaN。
    """
    n = len(close)
    atr = np.full(n, np.nan, dtype=np.float32)

    if n < window + 1:
        return atr

    prevClose = close[:-1]

    hl = high[1:] - low[1:]
    hpc = np.abs(high[1:] - prevClose)
    lpc = np.abs(low[1:] - prevClose)

    tr = np.maximum(hl, np.maximum(hpc, lpc)).astype(np.float32)

    firstAtr = tr[:window].mean()
    alpha = 1.0 / window

    wilder = np.empty(len(tr), dtype=np.float32)
    wilder[window - 1] = firstAtr

    for i in range(window, len(tr)):
        wilder[i] = wilder[i - 1] * (1.0 - alpha) + tr[i] * alpha

    atr[window:] = wilder[window - 1:]

    return atr


def buildFeatures(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> dict[str, np.ndarray]:
    closeSeries = pd.Series(close, copy=False)

    atrRaw = calcAtr(high, low, close, window=14)

    ma5 = closeSeries.rolling(5).mean().to_numpy(dtype=np.float32)
    ma20 = closeSeries.rolling(20).mean().to_numpy(dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        atrPct = (atrRaw / close.astype(np.float32)).astype(np.float32)
        ma5OverMa20 = (ma5 / ma20).astype(np.float32)
        closeOverMa20 = (close.astype(np.float32) / ma20).astype(np.float32)

    return {
        "roc5": closeSeries.pct_change(5).to_numpy(dtype=np.float32),
        "roc20": closeSeries.pct_change(20).to_numpy(dtype=np.float32),
        "ma5": ma5,
        "ma20": ma20,
        "ma5_over_ma20": ma5OverMa20,
        "close_over_ma20": closeOverMa20,
        "volatility": closeSeries.pct_change().rolling(20).std().to_numpy(dtype=np.float32),
        "atr": atrRaw,
        "atr_pct": atrPct,
    }


@numba.njit
def _labelTriple(openArr, highArr, lowArr, atrArr, k, maxHoldDays):
    """
    標記 = 往後 maxHoldDays 天內是否觸及以下條件：

    target = entry + 2.0 * R
    stopLoss = entry - R

    label  1: 觸及 target，賺超過 2R
    label -1: 觸及 stopLoss，虧超過 1R
    label  0: 介於中間，排除不用
    """
    n = len(openArr)
    labels = np.full(n, -128, dtype=np.int8)

    for i in range(n - maxHoldDays):
        entry = openArr[i + 1]
        atrVal = atrArr[i]

        if np.isnan(entry) or np.isnan(atrVal) or atrVal <= 0:
            continue

        r = k * atrVal
        target = entry + 2.0 * r
        stopLoss = entry - r
        label = 0

        for j in range(maxHoldDays):
            idx = i + 1 + j
            h = highArr[idx]
            l = lowArr[idx]

            if np.isnan(h) or np.isnan(l):
                continue

            hitTp = h >= target
            hitSl = l <= stopLoss

            if hitTp and hitSl:
                label = -1
                break
            elif hitTp:
                label = 1
                break
            elif hitSl:
                label = -1
                break

        labels[i] = label

    return labels


@numba.njit(cache=True)
def _labelForwardReturn(closeArr, atrArr, holdDays=21):
    """
    標記 = 未來 holdDays 天的收益 / 進場日 ATR。

    label  1: risk-adjusted return > +1.0，賺超過 1 ATR
    label -1: risk-adjusted return < -0.5，虧超過 0.5 ATR
    label  0: 介於中間，排除不用
    """
    n = len(closeArr)
    labels = np.full(n, -128, dtype=np.int8)

    for i in range(n - holdDays - 1):
        entry = closeArr[i]
        future = closeArr[i + holdDays]
        atr = atrArr[i]

        if np.isnan(entry) or np.isnan(future) or np.isnan(atr) or atr <= 0:
            continue

        retOverAtr = (future - entry) / atr

        if retOverAtr > 1.0:
            labels[i] = 1
        elif retOverAtr < -0.5:
            labels[i] = -1
        else:
            labels[i] = 0

    return labels


@numba.njit(cache=True)
def _labelTripleWithTaiex(pack, packTAIEX, atrArr, k, maxHoldDays):
    """
    標記 = 往後 maxHoldDays 天內是否觸及以下條件：

    target = entry + 2.0 * R
    stopLoss = entry - R

    label  1: 觸及 target，賺超過 2R
    label -1: 觸及 stopLoss，虧超過 1R
    label  0: 介於中間，排除不用

    注意：
    目前此函式名稱保留 Taiex，但邏輯中尚未使用大盤資料。
    """
    openArr, highArr, lowArr = pack
    openArrTaiex, highArrTaiex, lowArrTaiex, closeArrTaiex = packTAIEX
    n = len(openArr)
    labels = np.full(n, -128, dtype=np.int8)

    for i in range(n - maxHoldDays):
        entry = openArr[i + 1]
        entryTaiex = openArrTaiex[i + 1]
        atrVal = atrArr[i]

        if np.isnan(entry) or np.isnan(entryTaiex) or np.isnan(atrVal) or atrVal <= 0:
            continue

        label = 0
        Tp = 2.0 * k * (atrVal / entry)
        Sl = -k * (atrVal / entry)

        for j in range(maxHoldDays):
            idx = i + 1 + j
            # 改成：都用各自的進場點計算累積報酬
            cumStock_H = highArr[idx] / entry - 1.0
            cumStock_L = lowArr[idx] / entry - 1.0
            cumTaiex = closeArrTaiex[idx] / entryTaiex - 1.0  # 用 close 更穩定

            relativeH = cumStock_H - cumTaiex  # 個股超額 high
            relativeL = cumStock_L - cumTaiex  # 個股超額 low

            hitTp = relativeH >= Tp   # Tp = 2 * k * atr_pct
            hitSl = relativeL <= Sl   # Sl = -k * atr_pct

            # perTaiexOpen = openArrTaiex[idx] / entryTaiex
            # h = highArr[idx]
            # l = lowArr[idx]

            # perH = h / entry
            # perL = l / entry
            # relativeH = perH - perTaiexOpen   # 個股 high 相對大盤當天 open
            # relativeL = perL - perTaiexOpen   # 個股 low  相對大盤當天 open
            # if np.isnan(h) or np.isnan(l) or np.isnan(perTaiexOpen):
            #     continue
            # hitTp = relativeH >= Tp
            # hitSl = relativeL <= Sl

            if hitTp and hitSl:
                label = -1
                break
            elif hitTp:
                label = 1
                break
            elif hitSl:
                label = -1
                break

        labels[i] = label

    return labels


def labelSample(
    df: pd.DataFrame,
    dataTaiex: pd.DataFrame,
    k: float = 1.0,
    maxHoldDays: int = 63
) -> pd.DataFrame:
    """
    對 df 的每一行，以隔天開盤作為進場價，
    往後最多 maxHoldDays 天，標記 1 / 0 / -1。

    Parameters
    ----------
    df:
        含 open, close, atr 欄位，已按日期升序排列的單支股票 DataFrame。

    dataTaiex:
        台股大盤資料 DataFrame。
        目前保留此參數，但標記邏輯尚未直接使用大盤資料。

    k:
        風險係數，R = k * ATR。

    maxHoldDays:
        最大持有天數，預設 63，約 3 個月。

    Returns
    -------
    df:
        加上 label 欄位。
        最後 maxHoldDays 筆因無足夠未來資料設為 NA。
    """

    df = df.copy()
    stockDates = pd.to_datetime(df["date"])
    taiexColumns = [
        "TAIEXopen",
        "TAIEXmax",
        "TAIEXmin",
        "TAIEXclose",
    ]
    alignedTaiex = (
        dataTaiex.assign(_dateKey=pd.to_datetime(dataTaiex["date"]))
        .drop_duplicates("_dateKey", keep="last")
        .set_index("_dateKey")
        .reindex(stockDates)[taiexColumns]
    )

    openArr = df["open"].to_numpy(dtype=np.float32)
    highArr = df["max"].to_numpy(dtype=np.float32)
    lowArr = df["min"].to_numpy(dtype=np.float32)
    atrArr = df["atr"].to_numpy(dtype=np.float32)
    pack = (openArr, highArr, lowArr)

    taiexOpenArr = alignedTaiex["TAIEXopen"].to_numpy(dtype=np.float32)
    taiexMaxArr = alignedTaiex["TAIEXmax"].to_numpy(dtype=np.float32)
    taiexMinArr = alignedTaiex["TAIEXmin"].to_numpy(dtype=np.float32)
    closeArrTaiex = alignedTaiex["TAIEXclose"].to_numpy(dtype=np.float32)
    packTAIEX = (taiexOpenArr, taiexMaxArr, taiexMinArr, closeArrTaiex)

    labels = _labelTripleWithTaiex(
        pack,
        packTAIEX,
        atrArr,
        k,
        maxHoldDays
    )

    labelSeries = pd.array(labels, dtype=pd.Int8Dtype())
    labelSeries[labels == -128] = pd.NA

    df["label"] = labelSeries

    return df


def dataProcess(dataTaiex, maxHoldDays: int = 63) -> pd.DataFrame:
    """
    計算每天所有股票的 RS Rating，並存成 cache/{today}_Data.pkl。
    """
    allStockScores = []

    for entry in os.scandir("data"):
        if not entry.is_file():
            continue

        fileName = entry.name

        if len(fileName) != 30:
            continue

        df = pd.read_pickle(entry.path)
        df = df[df["date"] >= "2020-01-01"]
        df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

        if len(df) <= 252:
            continue

        closeNp = df["close"].to_numpy(dtype=np.float64, copy=False)
        highNp = df["max"].to_numpy(dtype=np.float64, copy=False)
        lowNp = df["min"].to_numpy(dtype=np.float64, copy=False)

        weightedScore = calcWeightedScore(closeNp)
        validMask = ~np.isnan(weightedScore)

        if not validMask.any():
            continue

        features = buildFeatures(highNp, lowNp, closeNp)

        dateArr = df["date"].to_numpy()

        temp = pd.DataFrame({
            "stock_id": df["stock_id"].iloc[0],
            "date": dateArr[validMask],
            "volume": df["Trading_Volume"].to_numpy(dtype=np.int32, copy=False)[validMask],
            "volatility": features["volatility"][validMask],
            "weightedScore": weightedScore[validMask],
            "close": closeNp.astype(np.float32, copy=False)[validMask],
            "open": df["open"].to_numpy(dtype=np.float32, copy=False)[validMask],
            "max": highNp.astype(np.float32, copy=False)[validMask],
            "min": lowNp.astype(np.float32, copy=False)[validMask],
            "roc5": features["roc5"][validMask],
            "roc20": features["roc20"][validMask],
            "ma5": features["ma5"][validMask],
            "ma20": features["ma20"][validMask],
            "ma5_over_ma20": features["ma5_over_ma20"][validMask],
            "close_over_ma20": features["close_over_ma20"][validMask],
            "atr": features["atr"][validMask],
            "atr_pct": features["atr_pct"][validMask],
        })

        temp = temp.sort_values("date").reset_index(drop=True)

        temp = labelSample(
            temp,
            dataTaiex,
            k=1.5,
            maxHoldDays=maxHoldDays
        )

        allStockScores.append(temp)

    print(f"已處理 {len(allStockScores)} 支股票的 weightedScore 計算")

    if not allStockScores:
        print("沒有可用資料")
        return pd.DataFrame()

    bigDf = pd.concat(allStockScores, ignore_index=True)

    bigDf = bigDf.merge(
        dataTaiex,
        on="date",
        how="left"
    )

    print("正在計算同一天的 RS Rating 百分位排名...")

    bigDf["rsRating"] = (
        bigDf.groupby("date", sort=False)["weightedScore"]
        .rank(pct=True, method="min")
        .mul(100)
        .astype(np.uint8)
    )

    print("正在計算 ΔRS...")

    bigDf = bigDf.sort_values(["stock_id", "date"]).reset_index(drop=True)

    bigDf["deltaRS"] = (
        bigDf.groupby("stock_id", sort=False)["rsRating"]
        .transform(
            lambda x: calculateDeltaRs(
                x.to_numpy(dtype=np.float32)
            )
        )
    )

    bigDf["deltaRS_rank"] = (
        bigDf.groupby("date", sort=False)["deltaRS"]
        .transform(lambda x: x.rank(pct=True, method="min") * 100)
        .astype(np.float32)
    )

    bigDf = bigDf.sort_values(
        ["date", "rsRating"],
        ascending=[True, False],
        kind="mergesort"
    ).reset_index(drop=True)

    bigDf.drop(columns=["weightedScore"], inplace=True)

    os.makedirs("cache", exist_ok=True)

    today = bigDf["date"].max()

    bigDf.to_pickle(f"cache/{today}_Data.pkl")

    print(bigDf["label"].value_counts(dropna=False))

    return bigDf


def dataProcessTaiex():
    dfPath = None

    for entry in os.scandir("data"):
        if not entry.is_file():
            continue

        fileName = entry.name

        if fileName.endswith("TAIEX.pkl"):
            dfPath = entry.path
            break

    if dfPath is None:
        print("找不到 TAIEX 的資料檔")
        return None

    df = pd.read_pickle(dfPath)
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    openNp = df["open"].to_numpy(dtype=np.float64, copy=False)
    closeNp = df["close"].to_numpy(dtype=np.float64, copy=False)
    highNp = df["max"].to_numpy(dtype=np.float64, copy=False)
    lowNp = df["min"].to_numpy(dtype=np.float64, copy=False)

    features = buildFeatures(highNp, lowNp, closeNp)

    temp = pd.DataFrame({
        "date": df["date"],
        "TAIEXvolume": df["Trading_Volume"].to_numpy(dtype=np.int32, copy=False),
        "TAIEXvolatility": features["volatility"],
        "TAIEXopen": openNp.astype(np.float32, copy=False),
        "TAIEXclose": closeNp.astype(np.float32, copy=False),
        "TAIEXmax": highNp.astype(np.float32, copy=False),
        "TAIEXmin": lowNp.astype(np.float32, copy=False),
        "TAIEXroc5": features["roc5"],
        "TAIEXroc20": features["roc20"],
        "TAIEXma5": features["ma5"],
        "TAIEXma20": features["ma20"],
        "TAIEXma5_ratio": features["ma5_over_ma20"],
        "TAIEXtrend": features["roc5"] > features["roc20"],
        "TAIEXatr": features["atr"],
        "TAIEXatr_pct": features["atr_pct"],
    })

    return temp


def purgedWalkForward(
    df,
    nSplits=5,
    embargoDays=21,
    maxHoldDays=63
):
    """
    Purged Walk-Forward CV for time-series data.

    - Train / test 按時間切分
    - Purge: 移除 train 尾端與 test 重疊的 label 窗口
    - Embargo: test 開頭再多跳過 embargoDays
    """
    df["date"] = pd.to_datetime(df["date"])

    dates = df["date"].sort_values().unique()
    foldSize = len(dates) // (nSplits + 1)

    for i in range(nSplits):
        trainEnd = dates[(i + 1) * foldSize]
        testStart = dates[(i + 1) * foldSize + embargoDays]
        testEnd = dates[min((i + 2) * foldSize, len(dates) - 1)]

        purgeStart = trainEnd - pd.Timedelta(days=maxHoldDays * 1.5)

        train = df[df["date"] <= purgeStart]
        test = df[
            (df["date"] >= testStart)
            & (df["date"] <= testEnd)
        ]

        yield train.index, test.index


if __name__ == "__main__":
    dataProcess(dataProcessTaiex())
