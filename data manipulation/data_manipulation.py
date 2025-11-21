import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
# 4. Prepare data for all models
# -------------------

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

print('effeciency:', model_df["EFFICIENCY"].describe())


# Sequential split - used for time series data

n_samples = len(model_df)
train_idx = int(n_samples * 0.7)  # 70% for training
val_idx = int(n_samples * 0.85)   # 15% for validation (70% + 15% = 85%)
                                   # Remaining 15% for test

X_train = X.iloc[:train_idx]
X_val = X.iloc[train_idx:val_idx]
X_test = X.iloc[val_idx:]

y_train = y.iloc[:train_idx]
y_val = y.iloc[train_idx:val_idx]
y_test = y.iloc[val_idx:]

print(f"Total samples: {n_samples}")
print(f"Train: {len(X_train)} ({len(X_train)/n_samples*100:.1f}%)")
print(f"Validation: {len(X_val)} ({len(X_val)/n_samples*100:.1f}%)")
print(f"Test: {len(X_test)} ({len(X_test)/n_samples*100:.1f}%)")

print('\nEfficiency Statistics:')
print(f"Train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
print(f"Val   - Mean: {y_val.mean():.4f}, Std: {y_val.std():.4f}")
print(f"Test  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")

# -------------------
# 5. Train all models
# -------------------

# Create pipelines that include scaling
pipelines = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
}

# Train and evaluate (use unscaled X data)
for model_name, pipeline in pipelines.items():
    print(f"\nTraining {model_name}...")
    
    pipeline.fit(X_train, y_train)  # Pipeline handles scaling
    
    y_test_pred = pipeline.predict(X_test)
    
    print(f"Test R²: {r2_score(y_test, y_test_pred):.4f}")
    print(f"Test RMSE: {mean_squared_error(y_test, y_test_pred, squared=False):.4f}")


# -------------------
# 7. Evaluate all models
# -------------------

pipelines = {
    'Linear Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
    ])
}

# Store results and predictions
results = []
predictions = {}

# Train and evaluate each model (use unscaled X data)
for model_name, pipeline in pipelines.items():
    print(f"\n{'='*60}")
    print(f"Training {model_name}...")
    print('='*60)
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    
    # Predict on all sets
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)
    
    # Store predictions for plotting
    predictions[model_name] = {
        'train': y_train_pred,
        'val': y_val_pred,
        'test': y_test_pred
    }
    
    # Calculate metrics for each set
    for set_name, y_true, y_pred in [
        ('Train', y_train, y_train_pred),
        ('Validation', y_val, y_val_pred),
        ('Test', y_test, y_test_pred)
    ]:
        r2 = r2_score(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        
        results.append({
            'Model': model_name,
            'Set': set_name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        })
        
        print(f"\n{set_name} Set:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")

# Create results DataFrame for easy comparison
results_df = pd.DataFrame(results)

# Display summary tables
print("\n" + "="*80)
print("SUMMARY - R² Scores:")
print("="*80)
print(results_df.pivot(index='Model', columns='Set', values='R²').round(4))

print("\n" + "="*80)
print("SUMMARY - RMSE:")
print("="*80)
print(results_df.pivot(index='Model', columns='Set', values='RMSE').round(4))

print("\n" + "="*80)
print("SUMMARY - MAE:")
print("="*80)
print(results_df.pivot(index='Model', columns='Set', values='MAE').round(4))

# -------------------
# 8. Plot Actual vs Predicted Efficiency over Time
# -------------------

# Plot actual vs predicted for each model on TEST set
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (model_name, preds) in enumerate(predictions.items()):
    ax = axes[idx]
    
    # Get test predictions
    y_test_pred = preds['test']
    y_pred_series = pd.Series(y_test_pred, index=y_test.index)
    
    # Plot
    ax.plot(y_test.index, y_test, label="Actual", linewidth=1.5, alpha=0.8)
    ax.plot(y_pred_series.index, y_pred_series, label="Predicted", 
            linestyle="--", linewidth=1.5, alpha=0.8)
    
    # Get metrics for title
    test_results = results_df[(results_df['Model'] == model_name) & 
                              (results_df['Set'] == 'Test')]
    r2 = test_results['R²'].values[0]
    rmse = test_results['RMSE'].values[0]
    
    ax.set_title(f"{model_name}\nR²={r2:.4f}, RMSE={rmse:.4f}", fontsize=12)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Efficiency", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Create a comparison plot - all models on one graph
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Actual Efficiency", 
         linewidth=2, alpha=0.9, color='black')

colors = ['blue', 'green', 'red']
for idx, (model_name, preds) in enumerate(predictions.items()):
    y_test_pred = preds['test']
    y_pred_series = pd.Series(y_test_pred, index=y_test.index)
    
    plt.plot(y_pred_series.index, y_pred_series, 
             label=f"{model_name}", 
             linestyle="--", linewidth=1.5, alpha=0.7, color=colors[idx])

plt.xlabel("Time", fontsize=12)
plt.ylabel("Efficiency", fontsize=12)
plt.title("Model Comparison: Actual vs Predicted Solar Panel Efficiency", fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
