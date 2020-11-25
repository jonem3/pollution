import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse
from django.views.generic import TemplateView

about = TemplateView.as_view(template_name='explain.html')

def pollution(request):
    tabledata = []
    timestamps = []
    returndata = {}
    frcstpath = "val/wxfcs/all/json/{}"
    columns = ['wind gust',
               'temperature',
               'wind speed',
               'screen relative humidity',
               'weather type',
               'visibility']
    if request.method == "POST":
        jason = json.loads(request.body.decode('utf-8'))
        print(jason)
        locationid = jason['location']
        quality = jason['quality']

        #locationid = WeatherLocation.objects.filter(name=location).first().id
        dataurl = "{}{}?res=daily&key={}".format(settings.DATAPOINT_BASE_URL, frcstpath.format(locationid),
                                                 settings.DATAPOINT_API_KEY)
        r = requests.get(dataurl)
        status = r.status_code

        if 200 <= status < 300:
            data = json.loads(r.text)
            # print(data)
            for i in data['SiteRep']['DV']['Location']['Period']:
                for j in i['Rep']:
                    if j['$'] == 'Day':
                        date = i['value']
                        date += " 12:00"
                        time = datetime.strptime(date, "%Y-%m-%dZ %H:%M")
                        timestamps.append(time)
                        info = [int(j['Gn']), int(j['Dm']), int(j['S']), int(j['Hn']), int(j['W']), str(j['V'])]
                        tabledata.append(info)
            df = pd.DataFrame(np.array(tabledata), columns=columns, index=timestamps)
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
            for q in quality:
                qdata = {}
                model = tf.keras.models.load_model(str(q) + "_model")
                results = model.predict(df).flatten()
                for i in range(len(results)):
                    results[i] = int(round(results[i]))
                    qdata[str(timestamps[i].date())] = int(results[i])
                returndata[q] = qdata
    return JsonResponse(returndata)
