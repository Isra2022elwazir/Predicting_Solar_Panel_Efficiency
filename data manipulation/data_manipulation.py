import pandas as pd;


df1 = pd.read_csv("data manipulation/Plant_2_Generation_Data.csv")
df2 = pd.read_csv("data manipulation/Plant_2_Weather_Sensor_Data.csv")

merged_plant2 = pd.merge(df1, df2, on="DATE_TIME", how="inner")

merged_plant2.to_csv("data manipulation/merged_plant2.csv", index=False)