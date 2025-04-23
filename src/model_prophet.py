""" Import depndencies"""
import pandas as pd
from prophet import Prophet
import pickle

from os import listdir
from os.path import isfile, join

array_dfs = None
df_final = None
df_data_train = None
df_data_test = None
df_train_prophet = None
model_prophet = None

def train():
    extract_datasource()
    transform_dataframe()
    prepare_data()
    train_model()
    save_model()

def extract_datasource():
    """ Extract datasource """
    global array_dfs
    sources = [f for f in listdir("./datasets") if isfile(join("./datasets", f))]

    array_dfs = []

    for source in sources:
            if source != '.gitkeep':
                array_dfs.append(pd.read_csv(f"./datasets/{source}", low_memory=False))

def transform_dataframe():
    """ Transform Dataframe """
    global df_final
    df = pd.concat(array_dfs, ignore_index=True, sort=False)

    df_filtered = df[['Date', 'Consommation']]

    df_not_nan_consommation = df_filtered[df_filtered['Consommation'].notna()]

    df_sorted = df_not_nan_consommation.sort_values(by='Date')

    # aggregation calc mean of Consommation / day
    df_final = df_sorted.groupby('Date', as_index=False).agg({'Consommation': 'mean'})

def prepare_data():
    """ Prepare data """
    global df_data_train
    global df_data_test
    global df_train_prophet
    
    df_data_train = df_final[df_final['Date'] < "2020"]
    df_data_test = df_final[df_final['Date'] >= "2020"]

    df_train_prophet = df_data_train

    # date variable needs to be named "ds" for prophet
    df_train_prophet = df_train_prophet.rename(columns={"Date": "ds"})

    # target variable needs to be named "y" for prophet
    df_train_prophet = df_train_prophet.rename(columns={"Consommation": "y"})

def train_model():
    """ Train model """
    global model_prophet
    model_prophet = Prophet()
    model_prophet.fit(df_train_prophet)
    
def save_model():
    """ Save model """
    with open("./models/prophet_model.pkl", "wb") as f:
        pickle.dump(model_prophet, f)

def get_train_data():
    return df_data_train

def get_test_data():
    return df_data_test
