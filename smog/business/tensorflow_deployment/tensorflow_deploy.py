import json
import math
from datetime import datetime
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.backend import square, mean

import pandas as pd
from smog.models import WeatherLocation, PollutionLocation
from django.conf import settings
tabledata = []
columns = ['wind gust',
          'temperature',
          'wind speed',
          'screen relative humidity',
          'weather type',
          'max uv',
          'precipitation probability']
timestamps = []
locations = []

def loss_mse_warmup(y_true, y_pred):
    """
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    loss = tf.losses.mse(y_true_slice, y_pred_slice)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean

    :param y_true:
    :param y_pred:
    :return:
    """
    warmup_steps = 50
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]
    mse = mean(square(y_true_slice - y_pred_slice))
    return mse


model = keras.models.load_model("Bensonsave", custom_objects={'loss_mse_warmup': loss_mse_warmup})
frcstpath = "val/wxfcs/all/json/{}"



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
            locations.append(weather_name)


def get_data():
    location = input("Enter the location (testing purposes): ")
    locationid = ""
    for location in WeatherLocation.objects.filter(name=location).order_by("id"):
        locationid = location.id
    dataurl = "{}{}?res=daily&key={}".format(settings.DATAPOINT_BASE_URL, frcstpath.format(locationid),
                                               settings.DATAPOINT_API_KEY)
    r = requests.get(dataurl)
    status = r.status_code
    print(status)
    if 200 <= status < 300:
        print(r.text)
        data = json.loads(r.text)
        #print(data)
    pandadata = []
    for i in data['SiteRep']['DV']['Location']['Period']:
        for j in i['Rep']:
            if j['$'] == 'Day':
                date = i['value']
                date += " 12:00"
                time = datetime.strptime(date, "%Y-%m-%dZ %H:%M")
                timestamps.append(time)
                info = [int(j['Gn']), int(j['Dm']), int(j['S']), int(j['Hn']), int(j['W']), int(j['U']),
                        int(j['PPd'])]
            else:
                date = i['value']
                date += " 00:00"
                time = datetime.strptime(date, "%Y-%m-%dZ %H:%M")
                timestamps.append(time)
                info = [int(j['Gm']), int(j['Nm']), int(j['S']), int(j['Hm']), int(j['W']), 0, int(j['PPn'])]
            tabledata.append(info)
    df = pd.DataFrame(np.array(tabledata), columns=columns, index=timestamps)
    print(df.head())
    prediction = model.predict(df)
    print(prediction)