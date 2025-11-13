import pandas as pd


df1 = pd.read_csv("data manipulation/Plant_2_Generation_Data.csv")
df2 = pd.read_csv("data manipulation/Plant_2_Weather_Sensor_Data.csv")

merged_plant2 = pd.merge(df1, df2, on="DATE_TIME", how="inner")

merged_plant2.to_csv("data manipulation/merged_plant2.csv", index=False)

df = pd.read_csv("data manipulation/merged_plant2.csv")
df = df.drop(columns=["PLANT_ID_x", "SOURCE_KEY_x", "PLANT_ID_y", "SOURCE_KEY_y", "DAILY_YIELD", "TOTAL_YIELD"], errors="ignore")


target_column = "IRRADIATION"
df_cleaned = df[df[target_column] != 0.0]

agg_rules = {
    'DC_POWER': 'mean',
    'AC_POWER': 'mean',
    'AMBIENT_TEMPERATURE': 'mean',
    'MODULE_TEMPERATURE': 'mean',
    'IRRADIATION': 'mean'
}

df_cleaned['DATE_TIME'] = pd.to_datetime(df_cleaned['DATE_TIME'])
df_cleaned = df_cleaned.set_index('DATE_TIME')
df_30min = df_cleaned.resample('30Min').agg(agg_rules)

# Calculate efficiency
df_30min['EFFICIENCY'] = df_30min['DC_POWER'] / df_30min['IRRADIATION']

# Save the result to a new CSV file
df_30min.to_csv("data manipulation/plant2_efficiency.csv", index=True)

#When panels get much hotter than the air, efficiency drops (heat reduces voltage output).
#This feature captures thermal stress, which is an important factor in degradation.
df_30min["TEMP_DELTA"] = df_30min["MODULE_TEMPERATURE"] - df_30min["AMBIENT_TEMPERATURE"]

#Solar performance follows a daily pattern — low at sunrise/sunset, peak at noon. So adding hour helps
df_30min["HOUR"] = df_30min.index.hour

#Captures seasonal changes in sunlight, temperature, and weather.
df_30min["DOY"] = df_30min.index.dayofyear


df_30min.to_csv(r"data manipulation/merged_plant2_30min_with_efficiency.csv", index=True)
print("Saved: data manipulation/merged_plant2_30min_with_efficiency.csv")