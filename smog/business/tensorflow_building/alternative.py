import math
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

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
    df['weather type'] = df['weather type'].astype(int)
    return df, features


def learn():
    compare_locations()
    for location in WeatherLocation.objects.all().order_by("id"):
        if location.name in locations:
            raw_df, features_considered = build_tables(location.id)
            df = raw_df.copy()
            print(df.tail())
            print(df.isna().sum())
            df['weather type'] = df['weather type'].map({0: 'Clear night',
                                                         1: 'Sunny day',
                                                         2: 'Partly cloudy (night)',
                                                         3: 'Partly cloudy (day)',
                                                         4: 'Not used',
                                                         5: 'Mist',
                                                         6: 'Fog',
                                                         7: 'Cloudy',
                                                         8: 'Overcast',
                                                         9: 'Light rain shower (night)',
                                                         10: 'Light rain shower (day)',
                                                         11: 'Drizzle',
                                                         12: 'Light rain',
                                                         13: 'Heavy rain shower (night)',
                                                         14: 'Heavy rain shower (day)',
                                                         15: 'Heavy rain',
                                                         16: 'Sleet shower (night)',
                                                         17: 'Sleet shower (day)',
                                                         18: 'Sleet',
                                                         19: 'Hail shower (night)',
                                                         20: 'Hail shower (day)',
                                                         21: 'Hail',
                                                         22: 'Light snow shower (night)',
                                                         23: 'Light snow shower (day)',
                                                         24: 'Light snow',
                                                         25: 'Heavy snow shower (night)',
                                                         26: 'Heavy snow shower (day)',
                                                         27: 'Heavy snow',
                                                         28: 'Thunder shower (night)',
                                                         29: 'Thunder shower (day)',
                                                         30: 'Thunder'})
            df = pd.get_dummies(df, prefix='', prefix_sep='')
            print(df.tail())
            train_df = df.sample(frac=0.8, random_state=0)
            test_df = df.drop(train_df.index)
            sns.pairplot(train_df[['wind gust', 'temperature', 'wind speed', 'screen relative humidity', ]],
                         diag_kind='kde')
            plt.show()
            print(train_df.describe().transpose())
            air_qualities = ['air quality index NO2',
                             'air quality index O3',
                             'air quality index PM10',
                             'air quality index PM25',
                             'air quality index SO2']
            for aq in air_qualities:
                test_qualities(aq, train_df, test_df)


def test_qualities(aq, train_df, test_df):
    print(aq)
    train_features = train_df.copy()
    test_features = test_df.copy()
    train_labels = train_features.drop(columns=aq)
    test_labels = test_features.drop(columns=aq)

    normalizer = preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())
    first = np.array(train_features[:1])

    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())

    linear_model = tf.keras.Sequential([
        normalizer,
        layers.Dense(units=1)
    ])

    linear_model.predict(train_features[:10])
    print(linear_model.layers[1].kernel)

    linear_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    history = linear_model.fit(
        train_features, train_labels,
        epochs=100,
        # suppress logging
        verbose=0,
        # Calculate validation results on 20% of the training data
        validation_split=0.2)

    plot_loss(history, aq)

    test_results = {}

    test_results['linear_model'] = linear_model.evaluate(
        test_features, test_labels, verbose=0)


def plot_loss(history, aq):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel(('Error ' + aq))
    plt.legend()
    plt.grid(True)
    plt.show()
