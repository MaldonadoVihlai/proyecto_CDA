import numpy as np
import streamlit as st
import pandas as pd
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import pickle
import joblib
import requests
from datetime import date
from server import constants

@st.experimental_singleton()
def load_model():
    return joblib.load('server/model/' + constants.MODEL)
    # with open('model/' + constants.MODEL, 'rb') as pickle_file:
    #     content = pickle.load(pickle_file)
    #     return content


def get_currency_exchange():
    today = date.today().strftime('%m/%d/%Y')
    #currency = requests.get(constants.URL_CURRENCY.format(today, today)).json()
    #currency = float(currency['data'][0]['last_close'].replace(',', ''))
    currency = 4680
    return currency


def get_top_page_content(st):
    st.image(constants.IMAGE_BANNER)
    st.title(
        'Predicción del precio de un vehículo')
    st.markdown(
        '**Nota**: Complete el siguiente formulario para estimar el precio de su vehículo')

def apply_scaler(data_df):
    standard_scaler = StandardScaler()
    X_binary_encoding_min_max_df = pd.DataFrame(standard_scaler.fit_transform(data_df), columns = data_df.columns)
    return X_binary_encoding_min_max_df

class clean_cars:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_duplicates(self):
        self.dataframe = self.dataframe.drop(['ID'], axis=1).drop_duplicates()

    def replace_boolean_by_numeric_columns(self):
        self.dataframe['Engine volume'] = self.dataframe['Engine volume'].str.replace(
            ' Turbo', '').astype(float)

    # Limpieza de datos de acuerdo al primer EDA realizado.
    def cleaning_colums_original(self):
        self.dataframe['Mileage'] = self.dataframe['Mileage'].str.replace(
            " km", "").astype(int)
        self.dataframe[self.dataframe['Levy'] == '-']
        self.dataframe['Levy'] = self.dataframe['Levy'].str.replace(
            '-', "0").astype(int)
        self.dataframe['Doors'] = self.dataframe['Doors'].replace({
            "04-May": "4-5",
            "02-Mar": "2-3"})
        self.dataframe['Doors'] = self.dataframe['Doors'].replace(
            {'2-3': 1, '4-5': 2, '>5': 3})
        self.dataframe['Doors'] = self.dataframe['Doors'].astype(int)

    def cleaning_colums(self):
        self.dataframe['Mileage'] = self.dataframe['Mileage'].astype(int)
        self.dataframe['Doors'] = self.dataframe['Doors'].replace({
            '2-3':2,
            '4-5':4,
            '6 o mas':6,
            "5 o más": 5})
        self.dataframe['Doors'] = self.dataframe['Doors'].astype(int)

    # Se aplica codificación one-hot para las variables categóricas
    def one_hot_encoding(self):
        wheel_dummies = pd.get_dummies(
            self.dataframe['Wheel'], drop_first=True, prefix='Wheel_')
        self.dataframe = pd.concat(
            [self.dataframe, wheel_dummies], axis=1).drop('Wheel', axis=1)
        drive_wheels = pd.get_dummies(
            self.dataframe['Drive wheels'], prefix='Drive wheels')
        self.dataframe = pd.concat(
            [self.dataframe, drive_wheels], axis=1).drop('Drive wheels', axis=1)
        gear_box_type = pd.get_dummies(
            self.dataframe['Gear box type'],  prefix='Gear_box_type')
        self.dataframe = pd.concat([self.dataframe, gear_box_type], axis=1).drop(
            'Gear box type', axis=1)
        fuel_type = pd.get_dummies(
            self.dataframe['Fuel type'], prefix='fuel_type_')
        self.dataframe = pd.concat(
            [self.dataframe, fuel_type], axis=1).drop('Fuel type', axis=1)
        self.dataframe['Leather interior'] = self.dataframe['Leather interior'].replace({
                                                                                        'Yes': 1, 'No': 0})
        category = pd.get_dummies(
            self.dataframe['Category'], drop_first=True, prefix='category_')
        self.dataframe = pd.concat(
            [self.dataframe, category], axis=1).drop('Category', axis=1)

    def clean_dataframe(self):
        self.replace_boolean_by_numeric_columns()
        self.cleaning_colums()
        return self.dataframe

    def clean_dataframe_original(self):
        self.replace_boolean_by_numeric_columns()
        self.cleaning_colums_original()
        return self.dataframe.drop(['ID', 'Price'], axis=1)

    def encode_one_hot_dataframe(self):
        self.one_hot_encoding()
        return self.dataframe

    # Se utilizará codificación binaria para variables con alta cardinalidad como Manufacturer y Model
    def encode_binary_dataframe(self):
        self.dataframe = ce.BinaryEncoder().fit_transform(self.dataframe)
        return self.dataframe
