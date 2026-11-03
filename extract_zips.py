"""
extract_zips.py
Turns Sensor Logger zip exports into clean, labelled CSV files.

Input : raw_zips/<activity>_<NN>.zip   e.g. walking_03.zip
Output: data/train/<activity>_<NN>.csv  (NN 01-12)
        data/test/<activity>_<NN>.csv   (NN 13 and above -> unseen test data)

Each output file has these columns:
    time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
(time_s starts at 0 and is on a uniform 50 Hz grid)

Run:  python prepare_data.py
"""
import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

RAW_DIR = "raw_zips"
OUT_DIR = "data"
TARGET_HZ = 50          # common sampling rate for every recording
TRIM_SECONDS = 1.0      # remove first/last second (getting into position)
TEST_FROM = 13          # files numbered >= 13 become the unseen test set
ACTIVITIES = ["standing", "walking", "jumping", "still"]


def read_sensor(zf, name, prefix):
    """Read one sensor CSV from the zip and rename x/y/z with a prefix."""
    with zf.open(name) as f:
        df = pd.read_csv(io.BytesIO(f.read()))
    df = df[["seconds_elapsed", "x", "y", "z"]].copy()
    df.columns = ["time_s", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    return df


def resample(df, hz):
    """Interpolate a sensor dataframe onto a uniform time grid at `hz`."""
    t0, t1 = df["time_s"].iloc[0], df["time_s"].iloc[-1]
    grid = np.arange(t0, t1, 1.0 / hz)
    out = pd.DataFrame({"time_s": grid})
    for col in df.columns[1:]:
        out[col] = np.interp(grid, df["time_s"].values, df[col].values)
    return out


def process_zip(path):
    with zipfile.ZipFile(path) as zf:
        acc = resample(read_sensor(zf, "Accelerometer.csv", "acc"), TARGET_HZ)
        gyr = resample(read_sensor(zf, "Gyroscope.csv", "gyro"), TARGET_HZ)

    # Align both sensors on the shared time range
    start = max(acc["time_s"].iloc[0], gyr["time_s"].iloc[0]) + TRIM_SECONDS
    end = min(acc["time_s"].iloc[-1], gyr["time_s"].iloc[-1]) - TRIM_SECONDS
    grid = np.arange(start, end, 1.0 / TARGET_HZ)

    out = pd.DataFrame({"time_s": grid - grid[0]})
    for df in (acc, gyr):
        for col in df.columns[1:]:
            out[col] = np.interp(grid, df["time_s"].values, df[col].values)
    return out


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.zip")))
    if not paths:
        print(f"No zip files found in '{RAW_DIR}/'")
        return

    for sub in ("train", "test"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    summary = []
    for path in paths:
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"^(standing|walking|jumping|still)_(\d+)$", base)
        if not m:
            print(f"SKIPPED (bad name, use e.g. walking_03.zip): {base}.zip")
            continue
        activity, num = m.group(1), int(m.group(2))
        df = process_zip(path)
        split = "test" if num >= TEST_FROM else "train"
        out_path = os.path.join(OUT_DIR, split, f"{activity}_{num:02d}.csv")
        df.to_csv(out_path, index=False)
        summary.append((activity, split, len(df) / TARGET_HZ))
        print(f"{out_path:35s} {len(df):4d} rows  {len(df)/TARGET_HZ:5.1f}s")

    # Totals per activity (rubric wants >= 90 s per activity)
    s = pd.DataFrame(summary, columns=["activity", "split", "seconds"])
    print("\nTotal clean seconds per activity and split:")
    print(s.pivot_table(index="activity", columns="split",
                        values="seconds", aggfunc="sum").round(1))
    print("\nNumber of files per activity:")
    print(s.groupby("activity").size())


if __name__ == "__main__":
    main()