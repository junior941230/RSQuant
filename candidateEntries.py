import pandas as pd


def processCandidateEntries(df):
    filteredDf = df[df["rsRating"] > 87]
    # 把entryPrice是NaN的行过滤掉
    filteredDf = filteredDf[filteredDf["entryPrice"].notna()]
    # 把weightedScore刪掉
    filteredDf = filteredDf.drop(columns=["weightedScore"])
    # filteredDf["daysSinceFirstSignal"] = 
    return filteredDf


if __name__ == "__main__":
    df = pd.read_pickle("RsRatingServer/cache/2026-03-27_RS.pkl")
    filteredDf = processCandidateEntries(df)
    print(filteredDf)
