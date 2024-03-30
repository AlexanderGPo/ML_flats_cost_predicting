import json

import numpy as np
import pandas as pd

from sklearn import ensemble

from urllib.request import urlopen
from urllib.parse import quote

EPS_FOR_FINDING_NEIGHBOURS = 0.005
API_KEY = '6601d83a5897f708562303yvp54b18e'
AUTHOR_TYPE_OPTIONS = ['homeowner', 'official_representative', 'real_estate_agent', 'realtor',
                       'representative_developer']


class PredictingModel:
    """ Model, created with RandomForestRegressor, that gives you opportunity to predict flats prices, based on
    some extra information about particular flat. """

    def __init__(self):
        """ Initialization and training the model. Work with train-data. Counting some specific information,
        like "district quality". """
        global EPS_FOR_FINDING_NEIGHBOURS

        self.dataframe = pd.read_csv('data/data.csv', index_col=0)

        self.dataframe.loc[(self.dataframe['floor'] == self.dataframe['floors_count']), 'floor'] = -1
        self.dataframe.loc[(self.dataframe['floor'] == 1), 'floor'] = -1
        self.dataframe.loc[(self.dataframe['floor'] != 1), 'floor'] = 0

        self.dataframe = pd.get_dummies(self.dataframe, columns=['author_type'], drop_first=True)
        self.dataframe.drop(['author_type_unknown'], axis=1, inplace=True)

        self.dataframe['lat'] = self.dataframe['lat'].astype('float64')
        self.dataframe['lon'] = self.dataframe['lon'].astype('float64')

        self.dataframe['district_quality'] = 0
        self.dataframe['district_quality'] = self.dataframe['district_quality'].astype('float64')

        for current_line in range(np.shape(self.dataframe)[0]):
            try:
                current_lat = self.dataframe.iloc[current_line]['lat']
                current_lon = self.dataframe.iloc[current_line]['lon']
                nearest_flats_df = self.dataframe.loc[((abs(self.dataframe['lat'] - current_lat)
                                                        < EPS_FOR_FINDING_NEIGHBOURS)
                                                       & (abs(self.dataframe['lon'] - current_lon)
                                                          < EPS_FOR_FINDING_NEIGHBOURS))]
                average_price = ((nearest_flats_df['price'] / nearest_flats_df['total_meters']).sum()
                                 / np.shape(nearest_flats_df)[0])
                self.dataframe.at[self.dataframe.index.values.tolist()[current_line], 'district_quality'] \
                    = average_price
            except ZeroDivisionError:
                self.dataframe.at[self.dataframe.index.values.tolist()[current_line], 'district_quality'] = 0

        self.dataframe['quality_price'] = self.dataframe['total_meters'] * self.dataframe['district_quality']

        self.dataframe.fillna(value=0, inplace=True)

        self.x_data = self.dataframe.drop('price', axis=1)
        self.y_data = self.dataframe['price']

        self.model_RFR = ensemble.RandomForestRegressor()
        self.model_RFR.fit(self.x_data, self.y_data)

    def price_predicting(self, predicting_series: pd.Series) -> int:
        """ Function, that allows you to use trained model. Using pd.Series with data about flat,
        it returns predicted cost (rubles). """
        url = f'https://geocode.maps.co/search?q={quote(predicting_series['address'])}&api_key={API_KEY}'
        jsonurl = urlopen(url)
        geocoder_output = json.loads(jsonurl.read())

        if len(geocoder_output) < 0:
            return -1

        predicting_series['lat'] = geocoder_output[0]['lat']
        predicting_series['lon'] = geocoder_output[0]['lon']
        predicting_series.drop('address', inplace=True)

        if predicting_series['floor'] == 1 or predicting_series['floor'] == predicting_series['floors_count']:
            predicting_series['floor'] = -1
        else:
            predicting_series['floor'] = 0

        for author_type_option in AUTHOR_TYPE_OPTIONS:
            predicting_series["author_type_" + author_type_option] = (
                    predicting_series['author_type'] == author_type_option)

        predicting_series.drop('author_type', inplace=True)

        current_lat = float(predicting_series['lat'])
        current_lon = float(predicting_series['lon'])

        nearest_flats_df = self.dataframe.loc[((abs(self.dataframe['lat'] - current_lat)
                                                < EPS_FOR_FINDING_NEIGHBOURS) & (abs(self.dataframe['lon'] -
                                                                                     np.full(len(self.dataframe['lon']),
                                                                                             current_lon))
                                                                                 < EPS_FOR_FINDING_NEIGHBOURS))]
        if np.shape(nearest_flats_df)[0] != 0:
            average_price = ((nearest_flats_df['price'] / nearest_flats_df['total_meters']).sum()
                             / np.shape(nearest_flats_df)[0])
        else:
            average_price = np.nan

        predicting_series['district_quality'] = average_price
        predicting_series['quality_price'] = (predicting_series['district_quality'] *
                                              predicting_series['total_meters'])

        variables = ['floor', 'floors_count', 'rooms_count', 'total_meters', 'lat', 'lon', 'author_type_homeowner',
                     'author_type_official_representative', 'author_type_real_estate_agent', 'author_type_realtor',
                     'author_type_representative_developer', 'district_quality', 'quality_price']

        data_dict = {}
        for i in variables:
            data_dict[i] = [predicting_series[i]]

        dataframe_predicting = pd.DataFrame.from_dict(data_dict)
        y_predicting = self.model_RFR.predict(dataframe_predicting)

        return y_predicting


model = PredictingModel()

info_for_predicting = pd.Series({'author_type': 'developer', 'floor': 15, 'floors_count': 43,
                                 'rooms_count': 3, 'total_meters': 77,
                                 'address': 'Армянский переулок, 2с2, Москва'})

print(model.price_predicting(info_for_predicting))
