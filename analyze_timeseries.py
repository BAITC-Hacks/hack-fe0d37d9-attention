import pandas as pd

FILES = {
    "T1": "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 1.csv",
    "T2": "Dataset HackAlemAI для участников 11.03.2023-28.02.2026 - turbine 2.csv",
}

TIME = "Статистическое время"
WIND = "Средняя скорость ветра(m/s)"
POWER = "Нормализованная активная мощность"
TEMP = "Средняя температура окружающей среды(°C)"


for name, file in FILES.items():

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    df = pd.read_csv(file)

    # Parse datetime
    df[TIME] = pd.to_datetime(df[TIME])

    df = df.sort_values(TIME)

    print("\nDATE RANGE")
    print("Start:", df[TIME].min())
    print("End:  ", df[TIME].max())

    # --------------------------------
    # Expected timestamps
    # --------------------------------

    expected = pd.date_range(
        start=df[TIME].min(),
        end=df[TIME].max(),
        freq="10min"
    )

    actual = pd.DatetimeIndex(df[TIME])

    missing_times = expected.difference(actual)

    print("\nTIMESTAMP ANALYSIS")
    print("Expected timestamps:", len(expected))
    print("Actual timestamps:  ", len(actual))
    print("Missing timestamps: ", len(missing_times))

    if len(missing_times):
        print("\nFirst missing timestamps:")
        print(missing_times[:20])

    # --------------------------------
    # Time intervals
    # --------------------------------

    diff = df[TIME].diff()

    print("\nMOST COMMON INTERVALS")
    print(diff.value_counts().head(10))

    print("\nLARGEST GAPS")
    gaps = (
        df.assign(gap=diff)
        .sort_values("gap", ascending=False)
        [[TIME, "gap"]]
        .head(15)
    )

    print(gaps.to_string(index=False))

    # --------------------------------
    # Power analysis
    # --------------------------------

    print("\nPOWER")

    print("Power = 0:",
          (df[POWER] == 0).sum(),
          f"({(df[POWER] == 0).mean()*100:.2f}%)")

    print("Power = 1:",
          (df[POWER] == 1).sum(),
          f"({(df[POWER] == 1).mean()*100:.2f}%)")

    # suspicious: strong wind but zero generation

    suspicious = df[
        (df[WIND] > 5) &
        (df[POWER] == 0)
    ]

    print("\nWind > 5 m/s but Power = 0:")
    print(len(suspicious))
    print(f"{len(suspicious)/len(df)*100:.2f}%")

    print("\nExamples:")
    print(
        suspicious[
            [TIME, WIND, POWER, TEMP]
        ].head(15).to_string(index=False)
    )

    # --------------------------------
    # Hourly aggregation
    # --------------------------------

    hourly = (
        df.set_index(TIME)
        .resample("1h")
        .agg({
            WIND: "mean",
            POWER: "mean",
            TEMP: "mean"
        })
    )

    print("\nHOURLY DATA")
    print("Rows:", len(hourly))

    print("\nHourly missing:")
    print(hourly.isna().sum())

    # --------------------------------
    # Power curve
    # --------------------------------

    print("\nPOWER CURVE")

    bins = pd.cut(
        df[WIND],
        bins=list(range(0, 26))
    )

    curve = (
        df.groupby(bins, observed=True)[POWER]
        .agg(["mean", "median", "count"])
    )

    print(curve.to_string())

    # --------------------------------
    # Monthly
    # --------------------------------

    df["month"] = df[TIME].dt.month

    monthly = df.groupby("month").agg({
        WIND: "mean",
        POWER: "mean",
        TEMP: "mean"
    })

    print("\nMONTHLY")
    print(monthly.round(3).to_string())