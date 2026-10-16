
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor

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

mh = PoissonRegressor(alpha=0.1, max_iter=1000)
ma = PoissonRegressor(alpha=0.1, max_iter=1000)

mh.fit(train[fitur], train["FTHG"])
ma.fit(train[fitur], train["FTAG"])

tim = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
tim_map = {t.lower(): t for t in tim}

def rata_rata(nama, konteks):
    data = df.iloc[:batas]
    hasil = []

    for _, row in data.iterrows():
        if konteks == "semua":
            if row["HomeTeam"] == nama:
                hasil.append((row["FTHG"], row["FTAG"]))
            elif row["AwayTeam"] == nama:
                hasil.append((row["FTAG"], row["FTHG"]))
        elif konteks == "kandang" and row["HomeTeam"] == nama:
            hasil.append((row["FTHG"], row["FTAG"]))
        elif konteks == "tandang" and row["AwayTeam"] == nama:
            hasil.append((row["FTAG"], row["FTHG"]))

    hasil = hasil[-5:]
    if not hasil:
        return 0.0, 0.0

    return (
        np.mean([x[0] for x in hasil]),
        np.mean([x[1] for x in hasil])
    )

def prediksi(home, away):
    if home not in tim_map or away not in tim_map:
        print("Nama tim tidak ditemukan.")
        return

    home = tim_map[home]
    away = tim_map[away]

    hc, hk = rata_rata(home, "semua")
    hck, hkk = rata_rata(home, "kandang")
    ac, ak = rata_rata(away, "semua")
    act, akt = rata_rata(away, "tandang")

    data = pd.DataFrame([[
        hc, hk, 0,
        hck, hkk, 0,
        ac, ak, 0,
        act, akt, 0
    ]], columns=fitur)

    lh = max(float(mh.predict(data)[0]), 0.05)
    la = max(float(ma.predict(data)[0]), 0.05)

    skor = []
    for h in range(9):
        for a in range(9):
            p = (
                math.exp(-lh) * lh**h / math.factorial(h)
                * math.exp(-la) * la**a / math.factorial(a)
            )
            skor.append((h, a, p))

    total = sum(x[2] for x in skor)
    menang = sum(p for h, a, p in skor if h > a) / total
    seri = sum(p for h, a, p in skor if h == a) / total
    kalah = sum(p for h, a, p in skor if h < a) / total
    under = sum(p for h, a, p in skor if h + a <= 2) / total

    print(f"\n{home} vs {away}")
    print(f"Estimasi gol: {lh:.2f} - {la:.2f}")
    print(f"Menang kandang: {menang*100:.1f}%")
    print(f"Seri: {seri*100:.1f}%")
    print(f"Menang tandang: {kalah*100:.1f}%")
    print(f"Under 2.5: {under*100:.1f}%")
    print(f"Over 2.5: {(1-under)*100:.1f}%")

    print("\n5 skor teratas:")
    for h, a, p in sorted(skor, key=lambda x: x[2], reverse=True)[:5]:
        print(f"{h}-{a}: {p/total*100:.2f}%")

print("=== FOOTBALL AI ===")
home = input("Tim kandang: ").strip().lower()
away = input("Tim tandang: ").strip().lower()
prediksi(home, away)
