
import pandas as pd
from pathlib import Path

files = sorted(Path("data").glob("E0*"))

for file in files:
    df = pd.read_csv(file)
    print(f"{file.name}: {len(df)} pertandingan")
