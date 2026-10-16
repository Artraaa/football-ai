import math
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, log_loss

df = pd.read_csv("data/fitur_performa_v2.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

fitur = [
    "home_gol_cetak_5",
    "home_gol_kebobolan_5",
    "home_poin_5",
    "home_gol_cetak_kandang_5",
    "home_gol_kebobolan_kandang_5",
    "home_poin_kandang_5",
    "away_gol_cetak_5",
    "away_gol_kebobolan_5",
    "away_poin_5",
    "away_gol_cetak_tandang_5",
    "away_gol_kebobolan_tandang_5",
    "away_poin_tandang_5",
]

batas = int(len(df) * 0.8)
train = df.iloc[:batas]
test = df.iloc[batas:]

mh = PoissonRegressor(alpha=1.0, max_iter=1000)
ma = PoissonRegressor(alpha=1.0, max_iter=1000)

mh.fit(train[fitur], train["FTHG"])
ma.fit(train[fitur], train["FTAG"])

ph = mh.predict(test[fitur])
pa = ma.predict(test[fitur])

print("MAE gol kandang:", round(mean_absolute_error(test["FTHG"], ph), 3))
print("MAE gol tandang:", round(mean_absolute_error(test["FTAG"], pa), 3))

def peluang(lh, la):
    skor = np.zeros((10, 10))
    for h in range(10):
        for a in range(10):
            skor[h, a] = (
                np.exp(-lh) * lh**h / math.factorial(h)
                * np.exp(-la) * la**a / math.factorial(a)
            )
    return [
        skor[np.tril_indices(10, -1)].sum(),
        np.trace(skor),
        skor[np.triu_indices(10, 1)].sum(),
    ]

prob = np.array([
    peluang(max(h, 0.001), max(a, 0.001))
    for h, a in zip(ph, pa)
])
prob /= prob.sum(axis=1, keepdims=True)

aktual = np.where(
    test["FTHG"].to_numpy() > test["FTAG"].to_numpy(), 0,
    np.where(test["FTHG"].to_numpy() == test["FTAG"].to_numpy(), 1, 2)
)

print("Akurasi 1X2:", round((prob.argmax(axis=1) == aktual).mean() * 100, 2), "%")
print("Log loss:", round(log_loss(aktual, prob, labels=[0, 1, 2]), 4))
