import numpy as np
import pandas as pd

#load raw auto-mpg.data
DATA_PATH = "../Original-Datasets/auto+mpg/auto-mpg.data"

col_names = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name"
]

#the .data file uses whitespace delimiter
df_raw = pd.read_csv(
    DATA_PATH,
    delim_whitespace=True,
    header=None,
    names=col_names,
    na_values=["?"]  #horsepower uses ? for missing
)

print("raw shape:", df_raw.shape)
print(df_raw.head())

#basic cleaning & type conversion
df = df_raw.copy()

#ensure numeric columns are numeric (horsepower may be object due to '?')
numeric_cols = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin"
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

#strip car_name whitespace
df["car_name"] = df["car_name"].astype(str).str.strip()

#missing values
print("\nmissing values per column:\n", df.isna().sum())

#drop rows with missing horsepower
df = df.dropna(subset=["horsepower"]).reset_index(drop=True)

print("\nafter dropping missing horsepower:", df.shape)

#categorical handling
#origin of car is categorical (1=USA, 2=Europe, 3=Japan) - keep it as categories for now then use dummy
df["origin"] = df["origin"].astype(int).astype("category")

#do dummy encoding learned in class
df_model = pd.get_dummies(df, columns=["origin"], prefix="origin", drop_first=True)

#final sanity checks
print("\nclean/model-ready shape:", df_model.shape)
print(df_model.dtypes)

#save preprocessed dataset
df_model.to_csv("autompg_preprocessed.csv", index=False)
print("Saved: autompg_preprocessed.csv.csv")