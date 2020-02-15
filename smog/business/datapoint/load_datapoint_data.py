from ...models import WeatherLocation
from ...models import WeatherObservation
from django.conf import settings
import requests
import json
from datetime import datetime

TOP_LEFT = (51.713675, -0.594479)
BOTTOM_RIGHT = (51.273194, 0.503083)
def store_observations(location, obs_path, location_current):
    obs_url = "{}{}&key={}".format(settings.DATAPOINT_BASE_URL, obs_path.format(location_current['id']),
                                   settings.DATAPOINT_API_KEY)
    obs_r = requests.get(obs_url)
    obs_status = obs_r.status_code
    if 200 <= obs_status < 300:
        obs_data = json.loads(obs_r.text)
        dv = (obs_data['SiteRep']['DV'])
        print(dv)
        for i in range(0, 2):

            for j in range(0, 24):
                if 'Location' in dv:
                    try:
                        try:
                            time = (dv['Location']['Period'][i]['value'])
                            time += " "
                        except:
                            pass
                        current_obs = (dv['Location']['Period'][i]['Rep'][j])
                        time += (current_obs['$'])
                        #print(type(time).__name__)
                        #print(time)
                        time = time.replace("Z", "")
                        times = time.split()
                        times[1] = int(times[1])
                        hours = times[1] // 60
                        #print(hours)
                        minutes = times[1] % 60
                        #print(minutes)
                        time = times[0] + " " + str(hours) + ":" + str(minutes)
                        #print(time)
                        time = datetime.strptime(time, '%Y-%m-%d %H:%M')
                        #print("After Conversion:", type(time))
                        try:
                            obs = WeatherObservation.objects.filter(
                                weather_location=location,
                                time_stamp=time
                            ).first()
                            print("Success!")
                        except:
                            obs = 0
                        print("obs =", obs)
                        if not obs:
                            print("Through")
                            obs = WeatherObservation()
                            obs.weather_location = location
                            obs.wind_direction = current_obs['D']

                            obs.time_stamp = time
                            obs.wind_gust = current_obs['G']
                            print("Wind Gust")
                            obs.temperature = current_obs['T']
                            print("1")
                            obs.wind_speed = current_obs['S']
                            print("2")
                            obs.pressure = current_obs['P']
                            print("3")
                            obs.pressure_tendency = current_obs['Pt']
                            print("4")
                            obs.dew_point = current_obs['Dp']
                            print("5")
                            obs.screen_relative_humidity = current_obs['H']
                            print("6")
                            obs.visibility = current_obs['V']
                            print("7")
                            obs.wind_direction = current_obs['D']
                            print("8")
                            #obs.weather_type = current_obs['W']
                            print("9")
                            #obs.max_uv = current_obs['U']
                            print("10")
                            #obs.precipitation_probability = current_obs['Pp']

                            print("All readings logged")
                            obs.save()
                            #print(obs)
                            print('Reading saved')
                        else:
                            print("Readings already present")
                    except:
                        pass
                else:
                    print("Error")

def load_data():
    print("Data Collection Starting...")
    site_list_path = "val/wxobs/all/json/sitelist"
    compatabilities_path = "val/wxobs/all/datatype/capabilities"
    obs_path = "val/wxobs/all/json/{}?res=hourly"

    site_list_url = "{}{}?key={}".format(settings.DATAPOINT_BASE_URL, site_list_path, settings.DATAPOINT_API_KEY)
    r = requests.get(site_list_url)
    status = r.status_code
    if 200<= status < 300:
        data = json.loads(r.text)
    else:
        print("Denied access - response {}".format(status))
    locations = []

    for location_current in data['Locations']['Location']:
        if BOTTOM_RIGHT[0] < float(location_current['latitude']) < TOP_LEFT[0] and \
                TOP_LEFT[1] < float(location_current['longitude']) < BOTTOM_RIGHT[1]:
            """
            location = WeatherLocation.objects.filter(
                id=location_current['id']
            ).first()
            if location is None:
            """
            locations.append(location_current)
            location = WeatherLocation()
            location.id = location_current['id']
            print("Location Added: ", location_current['id'])
            location.name = location_current['name']
            print("Location Name: ", location_current['name'])
            location.region = location_current['region']
            print("Location Region: ", location_current['region'])
            location.unitaryAuthArea = location_current['unitaryAuthArea']
            print("Location Unitary Auth Area: ", location_current['unitaryAuthArea'])
            location.latitude = location_current['latitude']
            print("Location Latitude: ", location_current['latitude'])
            location.longitude = location_current['longitude']
            print("Location Longitude: ", location_current['longitude'])
            location.elevation = location_current['elevation']
            print("Location Elevation: ", location_current['elevation'], "\n")
            location.save()
            print(location.id)
            store_observations(location, obs_path, location_current)
            """
            else:
                print("Already in database")
                store_observations(location, obs_path, location_current)
            """

