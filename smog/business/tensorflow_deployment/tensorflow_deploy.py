import json
import math
from datetime import datetime
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.backend import square, mean
import tensorflow as tf
import pandas as pd
from smog.models import WeatherLocation, PollutionLocation
from django.conf import settings
tabledata = []
columns = ['wind gust',
          'temperature',
          'wind speed',
          'screen relative humidity',
          'weather type',
           'visibility']
air_qualities = ['air quality index NO2',
                     'air quality index O3',
                     'air quality index PM10',
                     'air quality index PM25',
                     'air quality index SO2']
timestamps = []
locations = []
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
    location = WeatherLocation.objects.all()
    for i in location:
        loc = i.name
        test(loc)
def test(location):
    locationid = WeatherLocation.objects.filter(name=location).order_by("id").first().id
    dataurl = "{}{}?res=daily&key={}".format(settings.DATAPOINT_BASE_URL, frcstpath.format(locationid),
                                               settings.DATAPOINT_API_KEY)
    r = requests.get(dataurl)
    status = r.status_code
    print(status)
    if 200 <= status < 300:
        print(r.text)
        data = json.loads(r.text)

    for i in data['SiteRep']['DV']['Location']['Period']:
        for j in i['Rep']:
            if j['$'] == 'Day':
                date = i['value']
                date += " 12:00"
                time = datetime.strptime(date, "%Y-%m-%dZ %H:%M")
                print(time)
                timestamps.append(time)
                info = [int(j['Gn']), int(j['Dm']), int(j['S']), int(j['Hn']), int(j['W']), str(j['V'])]
                print(info)
                tabledata.append(info)

    print(tabledata)
    print(columns)
    print(timestamps)
    df = pd.DataFrame(np.array(tabledata), columns=columns, index=timestamps)
    print(df.head())
    df['visibility'] = df['visibility'].map({
        "UN": 0,
        "VP": 500,
        "PO": 2000,
        "MO": 9000,
        "GO": 15000,
        "VG": 30000,
        "EX": 40000
    })
    df = df.astype('float32')
    print(df.head())
    print(df.dtypes)

    for i in air_qualities:
        print(i)
        model = tf.keras.models.load_model(str(i) + "_model")
        results = model.predict(df).flatten()
        for i in range(len(results)):
            results[i] = int(round(results[i]))
        print(results)