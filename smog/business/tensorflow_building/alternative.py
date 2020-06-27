import math

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from smog.models import WeatherLocation, PollutionLocation, WeatherObservation, PollutionObservation

location_dict = {}

locations = []

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


def compare_locations():
    for location in WeatherLocation.objects.all().order_by("id"):
        closest = ""
        closest_dist = 1e15
        weather_id = location.id
        weather_name = location.name
        weather_lat = location.latitude
        weather_long = location.longitude
        for pollution_location in PollutionLocation.objects.all().order_by("site_code"):
            pollution_lat = pollution_location.latitude
            pollution_long = pollution_location.longitude
            distance = (3958 * 3.1415926 * math.sqrt(
                ((pollution_lat - weather_lat) ** 2) + (math.cos(weather_lat / 57.29578) ** 2) * (
                        (pollution_long - weather_long) ** 2)) / 180)
            if distance < closest_dist:
                closest_dist = distance
                closest = pollution_location.site_code
        if closest_dist <= 15:
            location_dict[weather_id] = closest
            locations.append(weather_name)


def build_tables(location_id):
    data = []
    timestamp = []
    columns = ['wind gust',
               'temperature',
               'wind speed',
               'screen relative humidity',
               'weather type',
               'max uv',
               'precipitation probability',
               'air quality index NO2',
               'air quality index O3',
               'air quality index PM10',
               'air quality index PM25',
               'air quality index SO2']
    air_qualities = ['air quality index NO2',
                     'air quality index O3',
                     'air quality index PM10',
                     'air quality index PM25',
                     'air quality index SO2']
    features = []
    for obs in WeatherObservation.objects.filter(weather_location=location_id).order_by("time_stamp"):
        if timestamp is not None and obs.time_stamp not in timestamp:
            timestamp.append(obs.time_stamp)
        if timestamp is not None:
            reading = [None, None, None, None, None, None, None, None, None, None, None, None]
            reading[0] = obs.wind_gust
            reading[1] = obs.temperature
            reading[2] = obs.wind_speed
            reading[3] = obs.screen_relative_humidity
            reading[4] = obs.weather_type
            reading[5] = obs.max_uv
            reading[6] = obs.precipitation_probability
            for pobs in PollutionObservation.objects.filter(time_stamp=obs.time_stamp.date(),
                                                            pollution_location=location_dict[location_id]).order_by(
                "time_stamp"):
                if pobs.species_code == "NO2":
                    reading[7] = pobs.air_quality_index
                    if "air quality index NO2" not in features:
                        features.append("air quality index NO2")
                elif pobs.species_code == "O3":
                    reading[8] = pobs.air_quality_index
                    if "air quality index O3" not in features:
                        features.append("air quality index O3")
                elif pobs.species_code == "PM10":
                    reading[9] = pobs.air_quality_index
                    if "air quality index PM10" not in features:
                        features.append("air quality index PM10")
                elif pobs.species_code == "PM25":
                    reading[10] = pobs.air_quality_index
                    if "air quality index PM25" not in features:
                        features.append("air quality index PM25")
                elif pobs.species_code == "SO2":
                    reading[11] = pobs.air_quality_index
                    if "air quality index SO2" not in features:
                        features.append("air quality index SO2")
                else:
                    pass
            data.append(reading)
    df = pd.DataFrame(data=data, columns=columns, index=timestamp)
    for i in air_qualities:
        if i not in features:
            df.drop(columns=[i])
    for i in features:
        df = df[df[i] != 0]
        df = df[df[i].notnull()]
    df.replace(to_replace=[None], value=np.nan, inplace=True)
    df = df.astype(float)
    df = df.interpolate(method='linear', axis=0, imit_direction='both')
    df.replace(to_replace=[np.nan], value=0, inplace=True)
    return df, features


def unvariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indicies = range(i - history_size, i)
        data.append(np.reshape(dataset[indicies], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def learn():
    compare_locations()
    for location in WeatherLocation.objects.all().order_by("id"):
        if location.name in locations:
            df, features_considered = build_tables(location.id)
            TRAIN_SPLIT = 1000
            tf.random.set_seed(13)
            #features_considered = ['air quality index NO2',
                                   #'air quality index O3',
                                   #'air quality index PM10',
                                   #'air quality index PM25',
                                   #'air quality index SO2']
            features = df[features_considered]
            # plt.plot(features)
            # plt.show()
            dataset = features.values
            data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
            data_std = dataset[:TRAIN_SPLIT].std(axis=0)
            dataset = (dataset - data_mean) / data_std
