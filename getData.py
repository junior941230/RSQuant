import requests
import pandas as pd
import io
import time


# response = requests.get("http://api.9tsai.xyz/rsRating")
# df = pd.read_pickle(io.BytesIO(response.content))
# print(df)
data = pd.read_pickle("cache/2026-04-21_Data.pkl")
print(data)
