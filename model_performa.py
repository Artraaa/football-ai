
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/fitur_performa.csv")
df = df.sort_values("Date").reset_index(drop=True)

fitur = [
    "home_gol_cetak_5",
    "home_gol_kebobolan_5",
    "home_poin_5",
    "away_gol_cetak_5",
    "away_gol_kebobolan_5",
    "away_poin_5",
]

df["Hasil"] = (
    (df["FTHG"] > df["FTAG"]).astype(int)
    - (df["FTHG"] < df["FTAG"]).astype(int)
)

X = df[fitur]
y = df["Hasil"]

# Pisahkan berdasarkan waktu, bukan acak
batas = int(len(df) * 0.8)
X_train, X_test = X.iloc[:batas], X.iloc[batas:]
y_train, y_test = y.iloc[:batas], y.iloc[batas:]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)
prediksi = model.predict(X_test)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))
print("Akurasi model:", round(accuracy_score(y_test, prediksi), 4))
print("Akurasi baseline:", round(baseline.score(X_test, y_test), 4))
print("\nLaporan evaluasi:")
print(classification_report(y_test, prediksi, zero_division=0))
