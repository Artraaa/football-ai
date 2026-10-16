
import pandas as pd
import numpy as np
from math import exp, factorial
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error

# Baca fitur performa 5 pertandingan terakhir
df = pd.read_csv("data/fitur_performa.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

fitur = [
    "home_gol_cetak_5",
    "home_gol_kebobolan_5",
    "home_poin_5",
    "away_gol_cetak_5",
    "away_gol_kebobolan_5",
    "away_poin_5",
]

# Pisahkan data berdasarkan waktu
batas = int(len(df) * 0.8)
train = df.iloc[:batas]
test = df.iloc[batas:]

# Latih model gol kandang dan tandang
model_home = PoissonRegressor(alpha=1.0, max_iter=1000)
model_away = PoissonRegressor(alpha=1.0, max_iter=1000)

model_home.fit(train[fitur], train["FTHG"])
model_away.fit(train[fitur], train["FTAG"])

pred_home = model_home.predict(test[fitur])
pred_away = model_away.predict(test[fitur])

print("MAE gol kandang:", round(
    mean_absolute_error(test["FTHG"], pred_home), 3
))
print("MAE gol tandang:", round(
    mean_absolute_error(test["FTAG"], pred_away), 3
))

# Fungsi probabilitas jumlah gol Poisson
def poisson_prob(lam, gol):
    return exp(-lam) * lam**gol / factorial(gol)

# Contoh pertandingan: ambil fitur terakhir yang tersedia
contoh = df.iloc[[-1]]
lambda_home = float(model_home.predict(contoh[fitur])[0])
lambda_away = float(model_away.predict(contoh[fitur])[0])

# Probabilitas skor 0 sampai 8 gol
hasil_skor = []

for gh in range(9):
    for ga in range(9):
        p = poisson_prob(lambda_home, gh) * poisson_prob(lambda_away, ga)
        hasil_skor.append((gh, ga, p))

hasil_skor.sort(key=lambda x: x[2], reverse=True)

p_home = sum(p for gh, ga, p in hasil_skor if gh > ga)
p_draw = sum(p for gh, ga, p in hasil_skor if gh == ga)
p_away = sum(p for gh, ga, p in hasil_skor if gh < ga)

print("\nContoh estimasi gol:")
print("Lambda kandang:", round(lambda_home, 2))
print("Lambda tandang:", round(lambda_away, 2))

print("\nProbabilitas hasil (perkiraan):")
print("Menang kandang:", round(p_home * 100, 1), "%")
print("Seri:", round(p_draw * 100, 1), "%")
print("Menang tandang:", round(p_away * 100, 1), "%")

print("\n3 skor paling mungkin:")
for gh, ga, p in hasil_skor[:3]:
    print(f"{gh}-{ga}: {p * 100:.2f}%")
# Evaluasi prediksi jumlah gol pada data testing
print("\nEvaluasi prediksi gol:")

print("Gol kandang aktual:", test["FTHG"].mean())
print("Gol kandang prediksi:", round(pred_home.mean(), 3))

print("Gol tandang aktual:", test["FTAG"].mean())
print("Gol tandang prediksi:", round(pred_away.mean(), 3))

# Evaluasi hasil pertandingan pada data testing
hasil_aktual = np.where(
    test["FTHG"] > test["FTAG"], 1,
    np.where(test["FTHG"] == test["FTAG"], 0, -1)
)

hasil_prediksi = np.where(
    pred_home > pred_away, 1,
    np.where(np.abs(pred_home - pred_away) < 0.25, 0, -1)
)

akurasi = np.mean(hasil_aktual == hasil_prediksi)

print("\nAkurasi hasil pertandingan:", round(akurasi * 100, 2), "%")

from sklearn.metrics import log_loss

# Evaluasi probabilitas pada seluruh data testing
probabilitas = []

for lh, la in zip(pred_home, pred_away):
    skor = np.zeros((9, 9))

    for gh in range(9):
        for ga in range(9):
            skor[gh, ga] = (
                poisson_prob(max(lh, 0.001), gh)
                * poisson_prob(max(la, 0.001), ga)
            )

    p_home = skor[np.tril_indices(9, -1)].sum()
    p_draw = np.trace(skor)
    p_away = skor[np.triu_indices(9, 1)].sum()

    total = p_home + p_draw + p_away
    probabilitas.append([
        p_home / total,
        p_draw / total,
        p_away / total
    ])

probabilitas = np.array(probabilitas)

# Label aktual: 0 = kandang, 1 = seri, 2 = tandang
aktual = np.where(
    test["FTHG"].to_numpy() > test["FTAG"].to_numpy(), 0,
    np.where(
        test["FTHG"].to_numpy() == test["FTAG"].to_numpy(),
        1, 2
    )
)

prediksi_1x2 = probabilitas.argmax(axis=1)

print("\n=== EVALUASI PROBABILITAS 1X2 ===")
print(
    "Akurasi:",
    round(np.mean(prediksi_1x2 == aktual) * 100, 2), "%"
)
print(
    "Log loss:",
    round(log_loss(aktual, probabilitas, labels=[0, 1, 2]), 4)
)

print("\nContoh probabilitas pertandingan testing pertama:")
print("Kandang:", round(probabilitas[0, 0] * 100, 1), "%")
print("Seri:", round(probabilitas[0, 1] * 100, 1), "%")
print("Tandang:", round(probabilitas[0, 2] * 100, 1), "%")
