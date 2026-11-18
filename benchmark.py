"""Engine-disjoint RUL benchmark at a fixed 100-cycle censoring point."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

SENSORS = [f"s{i}" for i in range(1, 22)]


def load(path):
    columns = ["unit", "cycle", "op1", "op2", "op3", *SENSORS]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python").dropna(axis=1, how="all")
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def nasa_score(y_true, y_pred):
    error = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0)).sum())


def score_predictions(y_capped, y_raw, prediction, baseline):
    """Report capped and uncapped cycle-100 errors separately."""
    y_capped = np.asarray(y_capped)
    y_raw = np.asarray(y_raw)
    prediction = np.asarray(prediction)
    baseline = np.asarray(baseline)
    return {
        "train_cycle_100_median_baseline_mae_cycles": float(mean_absolute_error(y_capped, baseline)),
        "model_mae_cycles": float(mean_absolute_error(y_capped, prediction)),
        "model_rmse_cycles": float(mean_squared_error(y_capped, prediction) ** 0.5),
        "model_nasa_asymmetric_score_on_capped_labels": nasa_score(y_capped, prediction),
        "model_raw_target_mae_cycles": float(mean_absolute_error(y_raw, prediction)),
        "median_baseline_raw_target_mae_cycles": float(mean_absolute_error(y_raw, baseline)),
        "test_engines_raw_rul_above_125": int((y_raw > 125).sum()),
        "test_engines_raw_rul_min_cycles": int(y_raw.min()),
        "test_engines_raw_rul_max_cycles": int(y_raw.max()),
    }


def main():
    source = Path("data/train_FD001.txt")
    df = load(source)
    max_cycle = df.groupby("unit").cycle.transform("max")
    df["raw_rul"] = max_cycle - df.cycle
    df["rul"] = df["raw_rul"].clip(upper=125)
    features = ["cycle", "op1", "op2", "op3", *SENSORS]
    for col in SENSORS:
        grouped = df.groupby("unit")[col]
        df[f"{col}_mean10"] = grouped.transform(lambda s: s.rolling(10, min_periods=1).mean())
        df[f"{col}_std10"] = grouped.transform(lambda s: s.rolling(10, min_periods=2).std()).fillna(0)
    features += [f"{col}_{stat}10" for col in SENSORS for stat in ("mean", "std")]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_rows, test_rows = next(splitter.split(df[features], df.rul, groups=df.unit))
    train, test = df.iloc[train_rows], df.iloc[test_rows]
    train_units, test_units = set(train.unit), set(test.unit)
    if train_units & test_units:
        raise RuntimeError("Engine IDs overlap between training and test.")

    cutoff = 100
    train_snap = train[train.cycle == cutoff]
    test_snap = test[test.cycle == cutoff]
    if train_snap.empty or test_snap.empty:
        raise SystemExit("No training or test engines reach the fixed 100-cycle censor point.")
    y_train = train_snap.rul
    y_test = test_snap.rul
    raw_y_test = test_snap.raw_rul
    model = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=0.8,
        n_jobs=-1, random_state=42,
    )
    model.fit(train[features], train.rul)
    prediction = model.predict(test_snap[features])
    baseline = np.full(len(y_test), float(y_train.median()))

    report = {
        "dataset": "NASA C-MAPSS Turbofan Engine Degradation Simulation, FD001",
        "simulation_scope": "Single operating condition, one simulated fault mode, run-to-failure engine trajectories.",
        "target": "Remaining cycles to simulated failure; model is trained on RUL capped at 125 cycles, then evaluated against both capped and uncapped cycle-100 labels",
        "evaluation": {
            "split": "Seeded engine-level 75/25 holdout; no engine IDs cross partitions.",
            "censor_cycle": cutoff,
            "train_engines": len(train_units),
            "test_engines": len(test_units),
            "train_rows_all_cycles": int(len(train)),
            "train_engines_scored_at_cycle_100": int(len(train_snap)),
            "test_engines_scored_at_cycle_100": int(len(test_snap)),
        },
        "features": features,
        "method": "300-tree RandomForestRegressor with trailing 10-cycle sensor means/stds; test prediction uses only measurements through cycle 100.",
        "test_metrics": score_predictions(y_test, raw_y_test, prediction, baseline),
        "source_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_archive_sha256": hashlib.sha256(Path("data/cmapss.zip").read_bytes()).hexdigest(),
        "limitations": [
            "This is simulator-generated data, not observed commercial fleet sensor or maintenance records.",
            "FD001 covers one operating condition and one simulated fault mode.",
            "The benchmark holds out engines from the same simulated subset; it is not an external-condition or external-fleet test.",
            "A single cycle-100 prediction is evaluated; no maintenance policy or alert threshold is assessed.",
            "The model is trained on capped targets; raw-target MAE is included to show the effect of scoring those capped predictions against uncapped RUL.",
            "Do not use for aviation maintenance decisions without validation on authorized real fleet data.",
        ],
    }
    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

