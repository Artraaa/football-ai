from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/gabungan_epl.csv")

df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

df["Hasil"] = df.apply(
    lambda row: 1 if row["FTHG"] > row["FTAG"]
    else 0 if row["FTHG"] == row["FTAG"]
    else -1,
    axis=1
)

fitur = pd.get_dummies(df[["HomeTeam", "AwayTeam"]])
target = df["Hasil"]

X_train, X_test, y_train, y_test = train_test_split(
    fitur, target, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42
)
model.fit(X_train, y_train)

prediksi = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, prediksi))
print("\nLaporan evaluasi:")
print(classification_report(y_test, prediksi, zero_division=0))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

print(
    "Akurasi baseline:",
    baseline.score(X_test, y_test)
)
