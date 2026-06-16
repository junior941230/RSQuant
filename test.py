import pandas as pd


def main():
    df = pd.read_pickle("cache/20260616_TrainingDataset.pkl")
    df = df[df["date"] >= "2024-01-01"]
    df = df[df["rsRating"] >= 87]
    df = df[df["label"].notna()]
    df = df[df["deltaRS"].notna()]
    print("現有欄位：", df.columns.tolist())
    nan_counts = df.isna().sum()
    if nan_counts.any():
        print(f"  [警告] df 含 NaN 欄位：\n{nan_counts[nan_counts > 0]}")
    print(f"篩選後資料筆數：{len(df)}")
    print(f"篩選後資料日期範圍：{df['date'].min()} ~ {df['date'].max()}")


if __name__ == "__main__":
    main()
