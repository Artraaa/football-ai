
import pandas as pd

df = pd.read_csv("data/gabungan_epl.csv")

df["Date"] = pd.to_datetime(
    df["Date"], dayfirst=True, errors="coerce"
)
df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")

df = df.dropna(
    subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
)
df = df.sort_values("Date").reset_index(drop=True)

riwayat = {}
fitur_baris = []

def statistik_tim(tim, konteks):
    data = riwayat.get(tim, [])

    if konteks != "semua":
        data = [x for x in data if x["konteks"] == konteks]

    data = data[-5:]

    if not data:
        return [0.0, 0.0, 0.0]

    return [
        sum(x["cetak"] for x in data) / len(data),
        sum(x["kebobolan"] for x in data) / len(data),
        sum(x["poin"] for x in data) / len(data),
    ]

# Semua laga pada tanggal yang sama memakai riwayat sebelumnya.
# Hasil laga hari itu baru dimasukkan setelah fiturnya dibuat.
for tanggal, pertandingan in df.groupby("Date", sort=True):

    baris_hari_ini = []

    for _, row in pertandingan.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        h = statistik_tim(home, "semua")
        hk = statistik_tim(home, "kandang")
        a = statistik_tim(away, "semua")
        at = statistik_tim(away, "tandang")

        baris_hari_ini.append({
            "Date": tanggal,
            "HomeTeam": home,
            "AwayTeam": away,
            "home_gol_cetak_5": h[0],
            "home_gol_kebobolan_5": h[1],
            "home_poin_5": h[2],
            "home_gol_cetak_kandang_5": hk[0],
            "home_gol_kebobolan_kandang_5": hk[1],
            "home_poin_kandang_5": hk[2],
            "away_gol_cetak_5": a[0],
            "away_gol_kebobolan_5": a[1],
            "away_poin_5": a[2],
            "away_gol_cetak_tandang_5": at[0],
            "away_gol_kebobolan_tandang_5": at[1],
            "away_poin_tandang_5": at[2],
            "FTHG": row["FTHG"],
            "FTAG": row["FTAG"],
        })

    fitur_baris.extend(baris_hari_ini)

    # Update riwayat setelah semua fitur tanggal ini selesai.
    for _, row in pertandingan.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        gh = int(row["FTHG"])
        ga = int(row["FTAG"])

        poin_home = 3 if gh > ga else 1 if gh == ga else 0
        poin_away = 3 if ga > gh else 1 if gh == ga else 0

        riwayat.setdefault(home, []).append({
            "konteks": "kandang",
            "cetak": gh,
            "kebobolan": ga,
            "poin": poin_home,
        })

        riwayat.setdefault(away, []).append({
            "konteks": "tandang",
            "cetak": ga,
            "kebobolan": gh,
            "poin": poin_away,
        })

hasil = pd.DataFrame(fitur_baris)
hasil.to_csv("data/fitur_performa_v2.csv", index=False)

print("Dataset berhasil diperbarui!")
print("Jumlah pertandingan:", len(hasil))
print("Jumlah kolom:", len(hasil.columns))
print("Rentang tanggal:", hasil["Date"].min(), "-", hasil["Date"].max())
print(hasil.head())
