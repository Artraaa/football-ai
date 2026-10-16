
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error
)

# 1. LOAD DATA
df = pd.read_csv("data/fitur_performa_v2.csv")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

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

df = df.dropna(subset=fitur + ["Date", "FTHG", "FTAG"])
df = df.sort_values("Date").reset_index(drop=True)

df["Hasil"] = np.where(
    df["FTHG"] > df["FTAG"], 0,
    np.where(df["FTHG"] == df["FTAG"], 1, 2)
)

# Pisahkan berdasarkan tanggal agar urutan waktu terjaga
tanggal = np.sort(df["Date"].unique())
n = len(tanggal)

batas_train = int(n * 0.70)
batas_validasi = int(n * 0.85)

tgl_train = tanggal[:batas_train]
tgl_validasi = tanggal[batas_train:batas_validasi]
tgl_test = tanggal[batas_validasi:]

train = df[df["Date"].isin(tgl_train)].copy()
validasi = df[df["Date"].isin(tgl_validasi)].copy()
test = df[df["Date"].isin(tgl_test)].copy()

print("Data train:", len(train))
print("Data validasi:", len(validasi))
print("Data test:", len(test))

# 2. FUNGSI PROBABILITAS SKOR POISSON

def probabilitas_1x2(lh, la):
    skor = np.zeros((15, 15))

    for h in range(15):
        for a in range(15):
            skor[h, a] = (
                math.exp(-lh) * lh**h / math.factorial(h)
                * math.exp(-la) * la**a / math.factorial(a)
            )

    
    
    menang = np.tril(skor, -1).sum()
    seri = np.trace(skor)
    kalah = np.triu(skor, 1).sum()

    hasil = np.array([menang, seri, kalah])
    print("DEBUG:", hasil / hasil.sum())
    return hasil / hasil.sum()



def prediksi_poisson(model_home, model_away, data):
    lh = np.maximum(model_home.predict(data[fitur]), 0.001)
    la = np.maximum(model_away.predict(data[fitur]), 0.001)

    prob = np.array([
        probabilitas_1x2(h, a)
        for h, a in zip(lh, la)
    ])

    return lh, la, prob

# 3. TUNING ALPHA MENGGUNAKAN DATA VALIDASI
print("\n=== TUNING ALPHA POISSON ===")

alpha_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
hasil_tuning = []

for alpha in alpha_list:
    mh = PoissonRegressor(alpha=alpha, max_iter=2000)
    ma = PoissonRegressor(alpha=alpha, max_iter=2000)

    mh.fit(train[fitur], train["FTHG"])
    ma.fit(train[fitur], train["FTAG"])

    _, _, prob = prediksi_poisson(mh, ma, validasi)

    loss = log_loss(
        validasi["Hasil"],
        prob,
        labels=[0, 1, 2]
    )

    hasil_tuning.append((alpha, loss))
    print(f"Alpha {alpha}: log loss {loss:.4f}")

alpha_terbaik, loss_validasi = min(
    hasil_tuning, key=lambda x: x[1]
)

print("\nAlpha terbaik:", alpha_terbaik)
print("Log loss validasi:", round(loss_validasi, 4))

# 4. LATIH ULANG DENGAN TRAIN + VALIDASI
data_latih = pd.concat([train, validasi]).sort_values("Date")

mh = PoissonRegressor(
    alpha=alpha_terbaik, max_iter=2000
)
ma = PoissonRegressor(
    alpha=alpha_terbaik, max_iter=2000
)

mh.fit(data_latih[fitur], data_latih["FTHG"])
ma.fit(data_latih[fitur], data_latih["FTAG"])

lh, la, prob_p = prediksi_poisson(mh, ma, test)

aktual = test["Hasil"].to_numpy()
pred_p = prob_p.argmax(axis=1)

# 5. RANDOM FOREST SEBAGAI PEMBANDING
rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

rf.fit(data_latih[fitur], data_latih["Hasil"])

prob_rf_asli = rf.predict_proba(test[fitur])
prob_rf = np.zeros((len(test), 3))

for i, kelas in enumerate(rf.classes_):
    prob_rf[:, int(kelas)] = prob_rf_asli[:, i]

pred_rf = prob_rf.argmax(axis=1)

# 6. EVALUASI TEST
print("\n=== HASIL AKHIR PADA DATA TEST ===")
print("Jumlah data test:", len(test))

print("\nPOISSON V2 TUNED")
print("MAE gol kandang:",
      round(mean_absolute_error(test["FTHG"], lh), 3))
print("MAE gol tandang:",
      round(mean_absolute_error(test["FTAG"], la), 3))
print("Akurasi 1X2:",
      round(accuracy_score(aktual, pred_p) * 100, 2), "%")
print("Log loss:",
      round(log_loss(aktual, prob_p, labels=[0, 1, 2]), 4))

print("\nRANDOM FOREST")
print("Akurasi 1X2:",
      round(accuracy_score(aktual, pred_rf) * 100, 2), "%")
print("Log loss:",
      round(log_loss(aktual, prob_rf, labels=[0, 1, 2]), 4))

print("\nEvaluasi selesai.")


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("\n=== DETAIL POISSON V2 ===")
print(classification_report(
    aktual,
    pred_p,
    labels=[0, 1, 2],
    target_names=["Menang kandang", "Seri", "Menang tandang"],
    zero_division=0
))

print("Confusion matrix Poisson:")
print(confusion_matrix(aktual, pred_p, labels=[0, 1, 2]))

print("\n=== DETAIL RANDOM FOREST ===")
print(classification_report(
    aktual,
    pred_rf,
    labels=[0, 1, 2],
    target_names=["Menang kandang", "Seri", "Menang tandang"],
    zero_division=0
))

print("Confusion matrix Random Forest:")
print(confusion_matrix(aktual, pred_rf, labels=[0, 1, 2]))

