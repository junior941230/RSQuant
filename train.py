import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, log_loss
from dataProcess import purged_walk_forward

FEATURE_COLS = [
    # ── 個股特徵 ──
    "volume", "rsRating", "deltaRS", "deltaRS_rank",
    "roc5", "roc20",
    "ma5_over_ma20", "close_over_ma20",
    "volatility", "atr_pct",

    # ── 大盤特徵 ──
    "TAIEXroc5", "TAIEXroc20",
    "TAIEXma5_ratio", "TAIEXvolatility",
    "TAIEXvolume", "TAIEXatr_pct",
]

TARGET_COL = "label"


def prepare_data(df: pd.DataFrame):
    """移除 label 為 NaN 的行，回傳乾淨的 X, y"""
    clean = df.dropna(subset=[TARGET_COL]).copy()
    clean[TARGET_COL] = clean[TARGET_COL].astype(int)
    X = clean[FEATURE_COLS]
    y = clean[TARGET_COL]
    return clean, X, y


def train_with_purged_wf(df: pd.DataFrame,
                         n_splits: int = 5,
                         max_hold_days: int = 21,
                         embargo_days: int = 5):
    """
    Purged Walk-Forward 訓練 + 評估
    """
    clean_df, X_all, y_all = prepare_data(df)

    fold_reports = []
    models = []

    for fold_i, (train_idx, test_idx) in enumerate(purged_walk_forward(clean_df, n_splits, max_hold_days, embargo_days)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i + 1}/{n_splits}")
        print(f"  Train: {len(train_idx):,} 筆")
        print(f"  Test:  {len(test_idx):,} 筆")
        X_train = X_all.loc[train_idx]
        y_train = y_all.loc[train_idx]
        X_test = X_all.loc[test_idx]
        y_test = y_all.loc[test_idx]

        # ── 類別分佈 ─────────────────────────────────
        print(f"  Train label 分佈: {dict(y_train.value_counts().sort_index())}")
        print(f"  Test  label 分佈: {dict(y_test.value_counts().sort_index())}")

        # ── CatBoost 訓練 ────────────────────────────
        train_pool = Pool(X_train, y_train)
        eval_pool = Pool(X_test, y_test)

        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            auto_class_weights="Balanced",
            eval_metric="MultiClass",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=100,
        )

        model.fit(train_pool, eval_set=eval_pool, use_best_model=True)

        # ── 評估 ─────────────────────────────────────
        y_pred = model.predict(X_test).flatten().astype(int)
        y_prob = model.predict_proba(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        logloss = log_loss(y_test, y_prob, labels=[-1, 0, 1])

        print(f"\n  Log Loss: {logloss:.4f}")
        print(classification_report(y_test, y_pred))

        fold_reports.append({
            "fold": fold_i + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "log_loss": logloss,
            "accuracy": report["accuracy"],
            "f1_macro": report["macro avg"]["f1-score"],
        })
        models.append(model)

    # ── 彙總 ─────────────────────────────────────────
    summary = pd.DataFrame(fold_reports)
    print("\n" + "=" * 60)
    print("Walk-Forward 彙總：")
    print(summary.to_string(index=False))
    print(
        f"\n平均 Log Loss:  {summary['log_loss'].mean():.4f} ± {summary['log_loss'].std():.4f}")
    print(
        f"平均 F1 (macro): {summary['f1_macro'].mean():.4f} ± {summary['f1_macro'].std():.4f}")

    return models, summary


if __name__ == "__main__":
    df = pd.read_pickle("cache/20260423_TrainingDataset.pkl")
    df = df[df["rsRating"] >= 87]
    print(df["label"].value_counts(dropna=False))
    train_with_purged_wf(df, n_splits=5, max_hold_days=21, embargo_days=5)
