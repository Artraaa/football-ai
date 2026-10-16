
import pandas as pd

df = pd.read_csv("data/gabungan_epl.csv")

print("TOTAL DATA:", len(df))
print("\nDATA KOSONG:")
print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].isna().sum())

print("\nJUMLAH HASIL PERTANDINGAN:")
print(df[["FTHG", "FTAG"]].describe())

print("\nCONTOH HASIL:")
print(df[["HomeTeam", "AwayTeam", "FTHG", "FTAG"]].head(10))
