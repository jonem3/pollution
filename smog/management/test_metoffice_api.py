import json
import requests
from django.conf import settings

TOP_LEFT = (51.713675, -0.594479)
BOTTOM_RIGHT = (51.273194, 0.503083)



def test_api():
    print("Hello, im here!")
    site_list_path = "val/wxobs/all/json/sitelist"
    capabilities_path = "val/wxobs/all/datatype/capabilities"
    obs_path = "val/wxobs/all/json/{}?res=hourly"

    site_list_url = "{}{}?key={}".format(settings.DATAPOINT_BASE_URL, site_list_path, settings.DATAPOINT_API_KEY)
    print("URL: {}".format(site_list_url))
    r = requests.get(site_list_url)
    status = r.status_code
    if 200 <= status < 300:
        data = json.loads(r.text)
    else:
        print("Denied access - response {}".format(status))
    locations = []

    for location in data['Locations']['Location']:
        if BOTTOM_RIGHT[0] < float(location['latitude']) < TOP_LEFT[0] and \
                TOP_LEFT[1] < float(location['longitude']) < BOTTOM_RIGHT[1]:
            locations.append(location)
            #print(locations)

            print(location['name'])
            obs_url = "{}{}&key={}".format(settings.DATAPOINT_BASE_URL, obs_path.format(location['id']), settings.DATAPOINT_API_KEY)
            print("URL: {}".format(obs_url))
            obs_r = requests.get(obs_url)
            obs_status = obs_r.status_code
            if 200 <= obs_status < 300:
                obs_data = json.loads(obs_r.text)
                #print(json.dumps(obs_data, indent=4))
                dv = (obs_data["SiteRep"]["DV"])
                print(dv["dataDate"])
                if "Location" in dv:
                    print(dv["Location"]["Period"][0]["Rep"][0])
            else:
                print("Error")

            #print(location['id'])
