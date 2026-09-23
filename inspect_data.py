import pandas as pd
from pathlib import Path

# ==========================================
# УКАЖИ НАЗВАНИЯ СВОИХ ДВУХ ФАЙЛОВ
# ==========================================

FILES = [
    "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 1.csv",
    "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv",
]


def inspect_dataset(file_path):
    print("\n" + "=" * 80)
    print(f"DATASET: {file_path}")
    print("=" * 80)

    # ---------- LOAD ----------
    df = pd.read_csv(file_path)

    print("\n[1] SHAPE")
    print(df.shape)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------- COLUMNS ----------
    print("\n[2] COLUMNS")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {repr(col)}")

    # ---------- SAMPLE ----------
    print("\n[3] FIRST 5 ROWS")
    print(df.head().to_string())

    print("\n[4] LAST 5 ROWS")
    print(df.tail().to_string())

    # ---------- DATA TYPES ----------
    print("\n[5] DATA TYPES")
    print(df.dtypes)

    # ---------- MISSING VALUES ----------
    print("\n[6] MISSING VALUES")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2)
    })

    print(missing.to_string())

    # ---------- DUPLICATES ----------
    print("\n[7] DUPLICATES")
    print("Duplicate rows:", df.duplicated().sum())

    # ---------- NUMERIC STATISTICS ----------
    print("\n[8] NUMERIC STATISTICS")
    print(df.describe(include="number").T.to_string())

    # ---------- UNIQUE VALUES ----------
    print("\n[9] UNIQUE VALUES")
    for col in df.columns:
        print(f"{col}: {df[col].nunique(dropna=False)}")

    # ---------- POSSIBLE DATETIME ----------
    print("\n[10] POSSIBLE DATETIME COLUMNS")

    for col in df.columns:
        if (
            "time" in col.lower()
            or "date" in col.lower()
            or "timestamp" in col.lower()
        ):
            parsed = pd.to_datetime(df[col], errors="coerce")

            valid = parsed.notna().sum()

            print(f"\nColumn: {col}")
            print(f"Parsed: {valid}/{len(df)}")

            if valid:
                print("Min:", parsed.min())
                print("Max:", parsed.max())

                sorted_time = parsed.dropna().sort_values()

                if len(sorted_time) > 1:
                    print(
                        "Most common interval:",
                        sorted_time.diff().value_counts().head()
                    )

    # ---------- CORRELATION ----------
    print("\n[11] CORRELATION")
    numeric = df.select_dtypes(include="number")

    if len(numeric.columns) > 1:
        print(numeric.corr().round(3).to_string())

    print("\n" + "=" * 80)


for file in FILES:
    path = Path(file)

    if not path.exists():
        print(f"\n❌ FILE NOT FOUND: {file}")
        continue

    try:
        inspect_dataset(path)
    except Exception as e:
        print(f"\n❌ ERROR reading {file}")
        print(e) 