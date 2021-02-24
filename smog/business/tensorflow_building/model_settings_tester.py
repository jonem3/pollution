import math
import os

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from tensorboard.plugins.hparams import api as hp

from pollution.settings import STATICFILES_DIRS
from smog.models import WeatherLocation, PollutionLocation, WeatherObservation, PollutionObservation

location_dict = {}

locations = []

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

hyperparameters = {'air quality index NO2': {
    'units_1': 64,
    'activation_1': 'relu',
    'units_2': 16,
    'activation_2': 'softmax',
    'optimizer': 'adam'
},
    'air quality index O3': {
        'units_1': 128,
        'activation_1': 'tanh',
        'units_2': 128,
        'activation_2': 'softmax',
        'optimizer': 'sgd'
    },
    'air quality index PM10': {
        'units_1': 64,
        'activation_1': 'sigmoid',
        'units_2': 32,
        'activation_2': 'softmax',
        'optimizer': 'sgd'
    },
    'air quality index PM25': {
        'units_1': 256,
        'activation_1': 'sigmoid',
        'units_2': 32,
        'activation_2': 'softmax',
        'optimizer': 'sgd'

    },
    'air quality index SO2': {
        'units_1': 256,
        'activation_1': 'tanh',
        'units_2': 32,
        'activation_2': 'softmax',
        'optimizer': 'sgd'
    }}


def compare_locations():
    for location in WeatherLocation.objects.all().order_by("id"):
        closest = ""
        closest_dist = 1e15
        weather_id = location.id
        weather_name = location.name
        weather_lat = location.latitude
        weather_long = location.longitude
        for pollution_location in PollutionLocation.objects.all().order_by("site_code"):
            R = 6372.8
            pollution_lat = pollution_location.latitude
            pollution_long = pollution_location.longitude
            distanceLat = math.radians(pollution_lat - weather_lat)
            distanceLong = math.radians(pollution_long - weather_long)
            weather_lat_1 = math.radians(weather_lat)
            pollution_lat_2 = math.radians(pollution_lat)

            a = math.sin(distanceLat / 2) ** 2 + math.cos(weather_lat_1) * math.cos(pollution_lat_2) * math.sin(
                distanceLong / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))

            distance = R * c
            if distance < closest_dist:
                closest_dist = distance
                closest = pollution_location.site_code
        if closest_dist <= 15:
            location_dict[weather_id] = closest
            locations.append(weather_name)


def build_tables2():
    count = 0
    data = []
    timestamp = []
    columns = ['wind gust',
               'temperature',
               'wind speed',
               'screen relative humidity',
               'weather type',
               # 'pressure',
               'visbility',
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
    for obs in WeatherObservation.objects.filter(weather_location_id__in=location_dict.keys(),
                                                 time_stamp__hour__in=(10, 11, 12, 13, 14)).order_by("time_stamp"):
        timestamp.append(count)
        count += 1
        if timestamp is not None:
            reading = [obs.wind_gust, obs.temperature, obs.wind_speed, obs.screen_relative_humidity, obs.weather_type,
                       obs.visibility, None, None, None, None, None]
            for pobs in PollutionObservation.objects.filter(time_stamp=obs.time_stamp.date(),
                                                            pollution_location=location_dict[
                                                                obs.weather_location_id]).order_by(
                "time_stamp"):
                if pobs.species_code == "NO2":
                    reading[6] = pobs.air_quality_index
                    if "air quality index NO2" not in features:
                        features.append("air quality index NO2")
                elif pobs.species_code == "O3":
                    reading[7] = pobs.air_quality_index
                    if "air quality index O3" not in features:
                        features.append("air quality index O3")
                elif pobs.species_code == "PM10":
                    reading[8] = pobs.air_quality_index
                    if "air quality index PM10" not in features:
                        features.append("air quality index PM10")
                elif pobs.species_code == "PM25":
                    reading[9] = pobs.air_quality_index
                    if "air quality index PM25" not in features:
                        features.append("air quality index PM25")
                elif pobs.species_code == "SO2":
                    reading[10] = pobs.air_quality_index
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
    df = df.astype('float32')
    df = df.interpolate(method='linear', axis=0, imit_direction='both')
    df.replace(to_replace=[np.nan], value=0, inplace=True)
    df['weather type'] = df['weather type'].astype('int64')
    return df, features


"""
def build_tables(location_id):
    data = []
    timestamp = []
    columns = ['wind gust',
               'temperature',
               'wind speed',
               'screen relative humidity',
               'weather type',
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
            reading = [obs.wind_gust, obs.temperature, obs.wind_speed, obs.screen_relative_humidity, obs.weather_type, None, None, None, None, None]
            for pobs in PollutionObservation.objects.filter(time_stamp=obs.time_stamp.date(),
                                                            pollution_location=location_dict[location_id]).order_by(
                "time_stamp"):
                if pobs.species_code == "NO2":
                    reading[5] = pobs.air_quality_index
                    if "air quality index NO2" not in features:
                        features.append("air quality index NO2")
                elif pobs.species_code == "O3":
                    reading[6] = pobs.air_quality_index
                    if "air quality index O3" not in features:
                        features.append("air quality index O3")
                elif pobs.species_code == "PM10":
                    reading[7] = pobs.air_quality_index
                    if "air quality index PM10" not in features:
                        features.append("air quality index PM10")
                elif pobs.species_code == "PM25":
                    reading[8] = pobs.air_quality_index
                    if "air quality index PM25" not in features:
                        features.append("air quality index PM25")
                elif pobs.species_code == "SO2":
                    reading[9] = pobs.air_quality_index
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
    df = df.astype('float32')
    df = df.interpolate(method='linear', axis=0, imit_direction='both')
    df.replace(to_replace=[np.nan], value=0, inplace=True)
    df['weather type'] = df['weather type'].astype('int64')
    return df, features
"""


def learn():
    compare_locations()
    # for location in WeatherLocation.objects.all().order_by("id"):
    # if location.name in locations:
    raw_df, features_considered = build_tables2()
    df = raw_df.copy()
    print(df.tail())
    print(df.isna().sum())

    """
     # Convert weather type to One-Hot
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
    """

    df = pd.get_dummies(df, prefix='', prefix_sep='')
    print(df.tail())
    print(df.dtypes)
    air_qualities = ['air quality index NO2',
                     'air quality index O3',
                     'air quality index PM10',
                     'air quality index PM25',
                     'air quality index SO2']

    features = ['wind gust',
                'temperature',
                'wind speed',
                'screen relative humidity',
                'weather type',
                'visbility']

    # Data split to make sure model works with unseen data
    train_dataset = df.sample(frac=0.9, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    # try:
    #     sns.pairplot(train_dataset[features], diag_kind="kde")
    #     plt.show()
    # except:
    #     pass

    trainable_qualities = train_dataset[air_qualities]
    testable_qualities = test_dataset[air_qualities]
    train_dataset.drop(air_qualities, axis=1, inplace=True)
    test_dataset.drop(air_qualities, axis=1, inplace=True)
    print(train_dataset.describe().transpose())

    normalizer = preprocessing.Normalization(dtype=float)
    normalizer.adapt(np.array(train_dataset))

    print(normalizer.mean.numpy())

    first = np.array(train_dataset[:1])
    with np.printoptions(precision=2, suppress=True):
        print('First example:', first)
        print()
        print('Normalized:', normalizer(first).numpy())
    for aq in air_qualities:
        train_labels = trainable_qualities.pop(aq)

        global HP_NUM_UNITS
        HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128, 256, 512]))

        global HP_NUM_UNITS_2
        HP_NUM_UNITS_2 = hp.HParam('num_units_2', hp.Discrete([64, 128, 256]))

        global HP_NUM_UNITS_3
        HP_NUM_UNITS_3 = hp.HParam('num_units_3', hp.Discrete([32, 64, 128]))

        global HP_NUM_UNITS_4
        HP_NUM_UNITS_4 = hp.HParam('num_units_4', hp.Discrete([16, 32, 64]))

        global METRIC_ACCURACY
        METRIC_ACCURACY = 'val_accuracy'

        global METRIC_ACCURACY_TEST
        METRIC_ACCURACY_TEST = 'accuracy_test'

        with tf.summary.create_file_writer(aq + '_logs/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[HP_NUM_UNITS,
                         HP_NUM_UNITS_2,
                         HP_NUM_UNITS_3,
                         HP_NUM_UNITS_4],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                         hp.Metric(METRIC_ACCURACY_TEST, display_name='Accuracy Test')],
            )

        session_num = 0

        for num_units in HP_NUM_UNITS.domain.values:
            for num_units_2 in HP_NUM_UNITS_2.domain.values:
                for num_units_3 in HP_NUM_UNITS_3.domain.values:
                    for num_units_4 in HP_NUM_UNITS_4.domain.values:
                        hparams = {HP_NUM_UNITS: num_units,
                                   HP_NUM_UNITS_2: num_units_2,
                                   HP_NUM_UNITS_3: num_units_3,
                                   HP_NUM_UNITS_4: num_units_4}
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run(aq + '_logs/hparam_tuning/' + run_name, hparams, normalizer, train_dataset,
                            train_labels, test_dataset, testable_qualities[aq])
                        session_num += 1


def train_test_model(hparams, normalizer, train_dataset, train_labels, test_dataset, testable_qualities):
    model = tf.keras.Sequential([
        normalizer,
        layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        layers.Dense(hparams[HP_NUM_UNITS_2], activation='relu'),
        layers.Dense(hparams[HP_NUM_UNITS_3], activation='relu'),
        layers.Dense(hparams[HP_NUM_UNITS_4], activation='softmax'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.Huber(),
                  metrics=['accuracy'])

    earlyStop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', min_delta=0.005, patience=100, verbose=0, mode='auto', restore_best_weights=True
    )

    callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.5,
                                                              min_lr=0,
                                                              patience=20,
                                                              verbose=0)

    model.fit(train_dataset, train_labels,
              epochs=1000,
              validation_split=0.2,
              callbacks=[earlyStop, callback_reduce_lr],
              verbose=0)
    _, accuracy = model.evaluate(test_dataset, testable_qualities)
    test_predictions = model.predict(test_dataset).flatten()
    total = 0
    correct = 0
    for i, j in zip(test_predictions, testable_qualities):
        if int(i) == int(j):
            correct += 1
        total += 1
    percentageCorrect = (correct / total) * 100
    print(str(percentageCorrect) + "% Accurate")
    print(str(accuracy) + "Accuracy Value")
    return accuracy, percentageCorrect


def run(run_dir, hparams, normalizer, train_dataset, train_labels, test_dataset, testable_qualities):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy, test_accuracy = train_test_model(hparams, normalizer, train_dataset, train_labels, test_dataset,
                                                   testable_qualities)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_ACCURACY_TEST, test_accuracy, step=1)
