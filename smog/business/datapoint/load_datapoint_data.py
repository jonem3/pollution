from ...models import WeatherLocation
from django.conf import settings
import requests
import json

TOP_LEFT = (51.713675, -0.594479)
BOTTOM_RIGHT = (51.273194, 0.503083)

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


