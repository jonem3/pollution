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
                                                 time_stamp__hour__in=(12, 13, 14, 15)).order_by("time_stamp"):
        timestamp.append(count)
        count += 1
        if timestamp is not None:
            reading = [obs.wind_gust, obs.temperature, obs.wind_speed, obs.screen_relative_humidity, obs.weather_type,
                       obs.visibility, None, None, None, None, None]
            for pobs in PollutionObservation.objects.filter(time_stamp=obs.time_stamp.date(),
                                                            pollution_location=location_dict[
                                                                obs.weather_location_id]).order_by("time_stamp"):
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



def learn():
    compare_locations()
    # for location in WeatherLocation.objects.all().order_by("id"):
    # if location.name in locations:
    raw_df, features_considered = build_tables2()
    df = raw_df.copy()
    print(df.tail())
    print(df.isna().sum())


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

    columns = ['wind gust',
               'temperature',
               'wind speed',
               'screen relative humidity',
               'weather type',
               'visbility',
               'air quality index NO2',
               'air quality index O3',
               'air quality index PM10',
               'air quality index PM25',
               'air quality index SO2']

    # Data split to make sure model works with unseen data
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)

    try:
        sns.pairplot(train_dataset[columns], diag_kind="kde")
        plt.show()
    except:
        pass

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
        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        print(linear_model.summary())
        print(linear_model.predict(train_dataset[:10]))

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoint.keras',
            save_weights_only=True,
            monitor='val_mean_absolute_error',
            mode='min',
            save_best_only=True,
            verbose=1
        )

        callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                  factor=0.5,
                                                                  min_lr=0,
                                                                  patience=20,
                                                                  verbose=1)

        callbacks = [model_checkpoint_callback]

        linear_model.compile(

            optimizer=tf.keras.optimizers.Adam(),
            loss='mean_absolute_error',
            metrics=['accuracy', tf.keras.metrics.MeanAbsoluteError()])

        history = linear_model.fit(
            train_dataset, train_labels,
            epochs=1000,
            verbose=1,
            validation_split=0.2,
            callbacks=callbacks)

        try:
            linear_model.load_weights('checkpoint.keras')
        except:
            print("ERROR LOADING WEIGHTS")

        plot_loss(history, aq)

        test_results = linear_model.evaluate(test_dataset, testable_qualities[aq], verbose=1)
        print(test_results)
        test_predictions = linear_model.predict(test_dataset).flatten()
        print(len(test_predictions))
        print(len(testable_qualities[aq]))
        a = plt.axes(aspect='equal')
        plt.scatter(testable_qualities[aq], test_predictions)
        plt.xlabel('True Values: ' + str(aq))
        plt.ylabel('Predictions: ' + str(aq))
        lims = [0, 10]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()

        error = test_predictions - testable_qualities[aq]
        plt.hist(error, bins=25)
        plt.xlabel('Prediction Error ' + str(aq))
        _ = plt.ylabel('Count')
        plt.show()
        modelname = os.path.join(STATICFILES_DIRS[0], (str(aq) + "_model"))
        linear_model.save(modelname)


def plot_loss(history, aq):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [' + aq + ']')
    plt.legend()
    plt.grid(True)
    plt.show()
