
import pandas as pd
from pathlib import Path

files = sorted(Path("data").glob("E0*"))
semua_data = []

for file in files:
    df = pd.read_csv(file)
    df["Sumber"] = file.name
    semua_data.append(df)

data = pd.concat(semua_data, ignore_index=True)

data = data.drop_duplicates(
    subset=["Date", "HomeTeam", "AwayTeam"]
)

data.to_csv("data/gabungan_epl.csv", index=False)

print("Dataset berhasil digabung!")
print("Total pertandingan:", len(data))
print("Kolom tersedia:", len(data.columns))
print(data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head())
