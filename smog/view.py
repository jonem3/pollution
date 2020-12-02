import json
import os
from datetime import datetime
from pollution.settings import STATICFILES_DIRS
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from django.conf import settings
from django.http import JsonResponse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt

about = TemplateView.as_view(template_name='explain.html')

band = {
    'air quality index NO2': {1: "0-67",
                              2: "68-134",
                              3: "135-200",
                              4: "201-267",
                              5: "268-334",
                              6: "335-400",
                              7: "401-467",
                              8: "468-534",
                              9: "535-600",
                              10: "601+"},
    'air quality index O3': {1: "0-33",
                             2: "34-66",
                             3: "67-100",
                             4: "101-120",
                             5: "121-140",
                             6: "141-160",
                             7: "161-187",
                             8: "188-213",
                             9: "214-240",
                             10: "241+"},
    'air quality index PM10': {1: "0-16",
                               2: "17-33",
                               3: "34-50",
                               4: "51-58",
                               5: "59-66",
                               6: "67-75",
                               7: "76-83",
                               8: "84-91",
                               9: "92-100",
                               10: "101+"},
    'air quality index PM25': {1: "0-11",
                               2: "12-23",
                               3: "24-35",
                               4: "36-41",
                               5: "42-47",
                               6: "48-53",
                               7: "54-58",
                               8: "59-64",
                               9: "65-70",
                               10: "71+"},
    'air quality index SO2': {1: "0-88",
                              2: "89-177",
                              3: "178-266",
                              4: "267-354",
                              5: "355-443",
                              6: "444-532",
                              7: "533-710",
                              8: "711-887",
                              9: "888-1064",
                              10: "1065+"}
}


@csrf_exempt
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
    if request.method == "GET":
        jason = json.loads(request.body.decode('utf-8'))
        print(jason)
        locationid = jason['location']
        quality = jason['quality']

        # locationid = WeatherLocation.objects.filter(name=location).first().id
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
                model = tf.keras.models.load_model(os.path.join(STATICFILES_DIRS[0], (str(q) + "_model")))
                results = model.predict(df).flatten()
                hsh = band[q]
                results = results.astype('U')
                for i in range(len(results)):
                    results[i] = round(float(results[i]))
                    if int(results[i]) < 1:
                        results[i] = "1"
                    elif int(results[i]) > 10:
                        results[i] = "10"
                    results[i] = str(results[i]) + "    (" + hsh[int(results[i])] + " µg/m<sup>3</sup>)"
                    qdata[str(timestamps[i].date())] = str(results[i])
                returndata[q] = qdata
    return JsonResponse(returndata)
