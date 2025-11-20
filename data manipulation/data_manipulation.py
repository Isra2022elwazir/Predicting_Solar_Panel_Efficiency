import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# -------------------
# 1. Load & merge data
# -------------------

data_dir = Path("data manipulation")

gen_path = data_dir / "Plant_2_Generation_Data.csv"
wth_path = data_dir / "Plant_2_Weather_Sensor_Data.csv"

# Parse DATE_TIME as datetime directly
df_gen = pd.read_csv(gen_path, parse_dates=["DATE_TIME"])
df_wth = pd.read_csv(wth_path, parse_dates=["DATE_TIME"])

# Inner join on DATE_TIME
df = pd.merge(df_gen, df_wth, on="DATE_TIME", how="inner")

# Drop columns we don't need
df = df.drop(
    columns=["PLANT_ID_x", "SOURCE_KEY_x", "PLANT_ID_y", "SOURCE_KEY_y", "DAILY_YIELD", "TOTAL_YIELD"],
    errors="ignore"
)

# -------------------
# 2. Clean & resample
# -------------------

# Remove zero or missing irradiance to avoid divide-by-zero
df_cleaned = df[(df["IRRADIATION"].notna()) & (df["IRRADIATION"] != 0.0)].copy()

# Set index to DATE_TIME and sort
df_cleaned = df_cleaned.set_index("DATE_TIME").sort_index()

# Aggregate to 30-minute intervals
agg_rules = {
    "DC_POWER": "mean",
    "AC_POWER": "mean",
    "AMBIENT_TEMPERATURE": "mean",
    "MODULE_TEMPERATURE": "mean",
    "IRRADIATION": "mean",
}

df_30min = df_cleaned.resample("30Min").agg(agg_rules)

# -------------------
# 3. Feature engineering
# -------------------

df_30min["IRRADIANCE_CORRECTED"] = df_30min["IRRADIATION"] * 4000
df_30min["EFFICIENCY"] = df_30min["DC_POWER"] / df_30min["IRRADIANCE_CORRECTED"]

print(df_30min["EFFICIENCY"].describe())

# Temperature difference (module - ambient)
df_30min["TEMP_DELTA"] = df_30min["MODULE_TEMPERATURE"] - df_30min["AMBIENT_TEMPERATURE"]

# Hour of day & day of year
df_30min["HOUR"] = df_30min.index.hour
df_30min["DOY"] = df_30min.index.dayofyear

# Power ratio (inverter performance)
df_30min["POWER_RATIO"] = df_30min["AC_POWER"] / df_30min["DC_POWER"]

# Save engineered dataset (optional)
out_csv = data_dir / "merged_plant2_30min_with_efficiency.csv"
df_30min.to_csv(out_csv, index=True)
print(f"Saved engineered dataset -> {out_csv}")

# -------------------
# 4. Prepare data for Linear Regression (chronological split + scaling)
# -------------------

# Clean infinities and NaNs (can appear from POWER_RATIO or EFFICIENCY)
model_df = df_30min.replace([np.inf, -np.inf], np.nan).dropna(
    subset=[
        "EFFICIENCY",
        "IRRADIATION",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "TEMP_DELTA",
        "HOUR",
        "DOY",
        "POWER_RATIO",
    ]
)

# Ensure sorted by time (chronological)
model_df = model_df.sort_index()

# Features (X) and target (y)
feature_cols = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "TEMP_DELTA",
    "HOUR",
    "DOY",
    "POWER_RATIO",
]
X = model_df[feature_cols]
y = model_df["EFFICIENCY"]

print( 'effeceincy: ', df_30min["EFFICIENCY"].describe())


# Chronological 80/20 split
n_samples = len(model_df)
split_idx = int(n_samples * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Total samples: {n_samples}, Train: {len(X_train)}, Test: {len(X_test)}")

# Standardize features (fit on train, transform both)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------
# 5. Train Linear Regression
# -------------------

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = lr_model.predict(X_test_scaled)

# -------------------
# 6. Evaluation metrics
# -------------------

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== Linear Regression (Chronological, with StandardScaler) ===")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

# Coefficients (on standardized features)
coef_table = pd.DataFrame(
    {
        "feature": feature_cols,
        "coefficient": lr_model.coef_,
    }
)
print("\nLinear Regression Coefficients (standardized features):")
print(coef_table)

# -------------------
# 7. Plot Actual vs Predicted Efficiency over Time
# -------------------

y_pred_series = pd.Series(y_pred, index=y_test.index)

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Actual Efficiency", linewidth=2)
plt.plot(y_pred_series.index, y_pred_series, label="Predicted Efficiency", linestyle="--", linewidth=2)

plt.xlabel("Time")
plt.ylabel("Efficiency")
plt.title("Actual vs Predicted Solar Panel Efficiency Over Time")

# --- Format x-axis for date + hour ---
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))  # show every 6 hours

plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

