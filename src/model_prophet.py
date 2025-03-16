""" Import depndencies"""
import pandas as pd
from prophet import Prophet
import pickle

from os import listdir
from os.path import isfile, join


""" Extract datasource """
sources = [f for f in listdir("./datasets") if isfile(join("./datasets", f))]

array_dfs = []

for source in sources:
        if source != '.gitkeep':
            array_dfs.append(pd.read_csv(f"./datasets/{source}", low_memory=False))

""" Transform Dataframe """
df = pd.concat(array_dfs, ignore_index=True, sort=False)

df_filtered = df[['Date', 'Consommation']]

df_not_nan_consommation = df_filtered[df_filtered['Consommation'].notna()]

df_sorted = df_not_nan_consommation.sort_values(by='Date')

# aggregation calc mean of Consommation / day
df_final = df_sorted.groupby('Date', as_index=False).agg({'Consommation': 'mean'})

""" Prepare data """
df_data_train = df_final[df_final['Date'] < "2020"]
df_data_test = df_final[df_final['Date'] >= "2020"]

df_train_prophet = df_data_train

# date variable needs to be named "ds" for prophet
df_train_prophet = df_train_prophet.rename(columns={"Date": "ds"})

# target variable needs to be named "y" for prophet
df_train_prophet = df_train_prophet.rename(columns={"Consommation": "y"})

""" Train model """
model_prophet = Prophet()
model_prophet.fit(df_train_prophet)

""" Save model """
with open("./models/prophet_model.pkl", "wb") as f:
    pickle.dump(model_prophet, f)
