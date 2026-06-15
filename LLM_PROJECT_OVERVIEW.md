# RSQuant 專案快速導覽

這份文件是給 LLM 快速理解專案用的，目標是讓模型在最短時間內掌握：這個專案在做什麼、資料怎麼流、核心檔案各自負責什麼、以及訓練與輸出依賴哪些欄位。

## 專案目的

RSQuant 是一個以台股個股與大盤資料為基礎的量化研究專案，主要流程是：

1. 從 `data/` 讀取個股與 TAIEX 原始資料。
2. 計算 RS Rating、ATR、均線、動能等特徵。
3. 依照未來報酬 / ATR 的規則產生三分類標籤 `label`。
4. 將整理好的訓練資料存進 `cache/`。
5. 用 CatBoost + Purged Walk-Forward 做時間序列訓練與評估。

## 核心資料流程

### 1. 原始資料

- 個股資料放在 `data/`。
- TAIEX 資料也放在 `data/`，檔名以 `TAIEX.pkl` 結尾。
- 原始欄位在程式中主要會用到：`date`、`stock_id`、`open`、`close`、`max`、`min`、`Trading_Volume`。

### 2. 特徵工程

主要在 [dataProcess.py](dataProcess.py) 中完成。

計算的特徵包含：

- `weightedScore`：以 3m / 6m / 9m / 12m 報酬加權得到的動能分數。
- `rsRating`：同一天所有股票的 `weightedScore` 百分位排名，轉成 0-100。
- `deltaRS`：對同一檔股票的 `rsRating` 做 rolling OLS 斜率，表示 RS 變化速度。
- `deltaRS_rank`：同一天內 `deltaRS` 的百分位排名。
- `roc5`、`roc20`：5 日 / 20 日漲跌幅。
- `ma5`、`ma20`、`ma5_over_ma20`、`close_over_ma20`：均線與價格相對均線位置。
- `volatility`：20 日報酬標準差。
- `atr`、`atr_pct`：ATR 與 ATR 相對價格比例。

### 3. 標籤產生

`labelSample()` 會根據未來 `maxHoldDays` 天內的價格行為產生三分類標籤：

- `1`：先碰到獲利目標。
- `-1`：先碰到停損條件。
- `0`：介於中間，保留為中性樣本。
- `NA`：未來資料不足，或 ATR / 價格無法計算。

目前標籤邏輯使用個股資料與大盤資料的相對超額報酬概念，實作在 `_labelTripleWithTaiex()`。

### 4. 訓練資料輸出

`dataProcess()` 最後會：

- 合併個股資料與 TAIEX 特徵。
- 依照日期做 `rsRating` 排名。
- 計算 `deltaRS` 與 `deltaRS_rank`。
- 依 `date` 與 `rsRating` 排序。
- 存成 `cache/{today}_Data.pkl`。

`main.py` 會再把 `dataProcess()` 的結果另存成 `cache/{YYYYMMDD}_TrainingDataset.pkl`。

## 檔案角色

### [main.py](main.py)

專案入口。執行資料處理主流程，輸出訓練資料 pickle。

### [dataProcess.py](dataProcess.py)

專案核心。包含：

- `calcWeightedScore()`：動能分數。
- `calculateDeltaRs()`：RS 變化斜率。
- `calcAtr()` 與 `buildFeatures()`：技術指標。
- `labelSample()`：產生訓練標籤。
- `dataProcessTaiex()`：整理大盤資料。
- `dataProcess()`：整合個股與大盤資料、生成完整訓練集。
- `purgedWalkForward()`：時間序列切分，避免資料洩漏。

### [train.py](train.py)

模型訓練與評估入口。使用：

- `FEATURE_COLS` 作為模型特徵。
- `label` 作為目標欄位。
- `purgedWalkForward()` 做時間序列交叉驗證。
- `CatBoostClassifier` 做三分類訓練。

### [candidateEntries.py](candidateEntries.py)

候選交易篩選工具。會從資料中挑出：

- `rsRating > 87` 的股票。
- `entryPrice` 非空的資料。
- 移除 `weightedScore` 欄位後輸出。

### [test.py](test.py)

目前比較像臨時檢查腳本，用來讀取 cache 資料與檢視欄位 / NaN 狀態。

## 訓練時最重要的欄位

`train.py` 會用這些特徵：

- 個股：`volume`、`rsRating`、`deltaRS`、`deltaRS_rank`、`roc5`、`roc20`、`ma5_over_ma20`、`close_over_ma20`、`volatility`、`atr_pct`
- 大盤：`TAIEXroc5`、`TAIEXroc20`、`TAIEXma5_ratio`、`TAIEXvolatility`、`TAIEXvolume`、`TAIEXatr_pct`

目標欄位是 `label`，類別為 `-1`、`0`、`1`。

## 時間序列切分

`purgedWalkForward()` 不是一般隨機切分，而是依日期做 walk-forward：

- 訓練集在前。
- 測試集在後。
- 會跳過一段 embargo，減少標籤窗口重疊造成的洩漏。
- 這對金融時間序列很重要，因為 `label` 本身會看未來 `maxHoldDays` 天。

## 需要注意的地方

- `cache/` 是主要中間產物資料夾。
- `data/` 必須有可讀的個股與 TAIEX pickle 檔。
- `train.py` 目前直接讀固定檔名 `cache/20260512_TrainingDataset.pkl`，實務上通常要改成最新輸出。
- `candidateEntries.py` 假設輸入資料已經有 `rsRating` 與 `entryPrice`。
- `test.py` 目前不是正式測試，而是檢查腳本。

## 一句話總結

這個專案的主線就是：從原始股價資料算出 RS / 技術特徵與三分類標籤，存成 cache 訓練集，再用時間序列切分訓練 CatBoost，最後提供候選交易篩選資料。
