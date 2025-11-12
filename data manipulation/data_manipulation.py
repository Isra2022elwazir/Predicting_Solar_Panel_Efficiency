import pandas as pd


df1 = pd.read_csv("data manipulation/Plant_2_Generation_Data.csv")
df2 = pd.read_csv("data manipulation/Plant_2_Weather_Sensor_Data.csv")

merged_plant2 = pd.merge(df1, df2, on="DATE_TIME", how="inner")

merged_plant2.to_csv("data manipulation/merged_plant2.csv", index=False)

df = pd.read_csv("data manipulation/merged_plant2.csv")

target_column = "IRRADIATION"
df_cleaned = df[df[target_column] != 0.0]

agg_rules = {
    'PLANT_ID_x': 'first',
    'SOURCE_KEY_x': 'first',
    'PLANT_ID_y': 'first',
    'SOURCE_KEY_y': 'first',
    'DC_POWER': 'mean',
    'AC_POWER': 'mean',
    'AMBIENT_TEMPERATURE': 'mean',
    'MODULE_TEMPERATURE': 'mean',
    'IRRADIATION': 'mean',
    'DAILY_YIELD': 'last',
    'TOTAL_YIELD': 'last'
}

df_cleaned['DATE_TIME'] = pd.to_datetime(df_cleaned['DATE_TIME'])
df_cleaned = df_cleaned.set_index('DATE_TIME')
df_30min = df_cleaned.resample('30Min').agg(agg_rules)