import json

import pickle

import numpy as np
import pandas as pd

from sklearn import ensemble

from urllib.request import urlopen
from urllib.parse import quote

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

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

        print("----------- Model initialization ----------- ")

        try:
            dbscan = pd.read_csv('model_data/data_for_dbscan.csv', index_col=0)
            self.dataframe = pd.read_csv('model_data/data_for_model.csv', index_col=0)
            self.model_RFR = pickle.load(open('model_data/model_RFR.sav', 'rb'))

            print("The model has already been created. Ready for work.")
        except FileNotFoundError:
            self.dataframe = pd.read_csv('train_data/data.csv', index_col=0)

            dataframe_dbscan = self.dataframe.drop(['floor', 'floors_count', 'lat', 'lon',
                                                    'author_type'], axis=1)
            dataframe_dbscan.to_csv('model_data/data_for_dbscan.csv')

            self.dataframe.drop(['url', 'district'], axis=1, inplace=True)

            self.dataframe.loc[(self.dataframe['floor'] == self.dataframe['floors_count']), 'floor'] = -1
            self.dataframe.loc[(self.dataframe['floor'] == 1), 'floor'] = -1
            self.dataframe.loc[(self.dataframe['floor'] != 1), 'floor'] = 0

            self.dataframe = pd.get_dummies(self.dataframe, columns=['author_type'], drop_first=True)
            self.dataframe.drop(['author_type_unknown'], axis=1, inplace=True)

            self.dataframe['lat'] = self.dataframe['lat'].astype('float64')
            self.dataframe['lon'] = self.dataframe['lon'].astype('float64')

            self.dataframe['district_quality'] = 0
            self.dataframe['district_quality'] = self.dataframe['district_quality'].astype('float64')

            print("Process: 1 / 3. Data has been successfully downloaded.")

            for current_line in range(np.shape(self.dataframe)[0]):
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

            self.dataframe['quality_price'] = self.dataframe['total_meters'] * self.dataframe['district_quality']

            print("Process: 2 / 3. District quality has been calculated.")

            self.dataframe.fillna(value=0, inplace=True)

            self.x_data = self.dataframe.drop('price', axis=1)
            self.y_data = self.dataframe['price']

            self.model_RFR = ensemble.RandomForestRegressor()
            self.model_RFR.fit(self.x_data, self.y_data)

            pickle.dump(self.model_RFR, open('model_data/model_RFR.sav', 'wb'))
            self.dataframe.to_csv('model_data/data_for_model.csv')

            print("Process: 3 / 3. Model has been saved.")

            print("The model has been created. Ready for work.")

        print("--- Model initialization has been ended. --- \n")

    def price_predicting(self, predicting_series: pd.Series, are_coordinates: bool = False) -> int:
        """ Function, that allows you to use trained model. Using pd.Series with data about flat,
        it returns predicted cost (rubles). """
        series_dbscan = predicting_series.copy()
        if not are_coordinates:
            url = f"https://geocode.maps.co/search?q={quote(predicting_series['address'])}&api_key={API_KEY}"
            jsonurl = urlopen(url)
            geocoder_output = json.loads(jsonurl.read())

            if len(geocoder_output) < 0:
                print("Warning: impossible to determine the coordinates. Accuracy can be reduced.")
                predicting_series['lat'] = 0
                predicting_series['lon'] = 0
            else:
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
        predicting_series['quality_price'] = (predicting_series['district_quality'] * predicting_series['total_meters'])

        y_predicting = self.model_RFR.predict(pd.DataFrame([predicting_series]))
        series_dbscan['price'] = y_predicting[0]

        self.find_similars(series_dbscan, are_coordinates)

        return y_predicting[0]

    def find_similars(self, series_dbscan: pd.Series, are_coordinates: bool = False) -> int:
        dataframe_urls = pd.read_csv('model_data/data_for_dbscan.csv', index_col=0)
        dataframe_dbscan = dataframe_urls.drop(['url', 'district'], axis=1)
        series_dbscan.drop(['floor', 'floors_count', 'address', 'author_type'], inplace=True)
        if are_coordinates:
            series_dbscan.drop(['lat', 'lon'], inplace=True)
        dataframe_dbscan = pd.concat([dataframe_dbscan, series_dbscan.to_frame().T])
        scaler = StandardScaler().fit(dataframe_dbscan)
        clustering = DBSCAN(eps=0.01).fit(scaler.transform(dataframe_dbscan))
        similar_proposals = []
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == clustering.labels_[-1] and clustering.labels_[-1] != -1:
                similar_proposals.append(dataframe_urls.iloc[i]['url'])
                if len(similar_proposals) > 5:
                    break

        if len(similar_proposals) > 0:
            print('Similar proposals:')
            print(*similar_proposals, sep='\n')

        else:
            print("Couldn't find any similar proposals.")
        print()
        return 0


model = PredictingModel()

info_for_predicting = pd.Series({'author_type': 'homeowner', 'floor': 20, 'floors_count': 37,
                                 'rooms_count': 5, 'total_meters': 120,
                                 'address': 'Москва, ул. Лобачевского, 120', 'lat': 55.774598, 'lon': 37.545950})

print(f"Predicted price: {model.price_predicting(info_for_predicting, True)}.")
