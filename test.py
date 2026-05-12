import pandas as pd

df = pd.read_pickle("cache/20260512_TrainingDataset.pkl")
df = df[df["date"] >= "2024-01-01"]
df = df[df["rsRating"] >= 87]
df = df[df["label"].notna()]
# 前向填充 rsRating 和 deltaRS 的 NaN 值
print("現有欄位：", df.columns.tolist())
df['rsRating'] = df['rsRating'].ffill()
df['deltaRS_rank'] = df['deltaRS_rank'].ffill()
df['deltaRS'] = df['deltaRS'].fillna(0)

nan_counts = df.isna().sum()
if nan_counts.any():
    print(f"  [警告] df 含 NaN 欄位：\n{nan_counts[nan_counts > 0]}")

