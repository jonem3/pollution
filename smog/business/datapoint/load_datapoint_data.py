from ...models import WeatherLocation
from ...models import WeatherObservation
from django.conf import settings
import requests
import json
from datetime import datetime
import logging
import pytz
from json import JSONDecodeError
from requests.exceptions import RequestException
logger = logging.getLogger('weather_log')

#TOP_LEFT = (51.713675, -0.594479)
TOP_LEFT = (52.6565051867658, -7.55715939721731)
#BOTTOM_RIGHT = (51.273194, 0.503083)
BOTTOM_RIGHT = (49.7667866669653, 1.04634518982586)


def store_observations(location, obs_path, location_current):
    obs_url = "{}{}&key={}".format(settings.DATAPOINT_BASE_URL, obs_path.format(location_current['id']),
                                   settings.DATAPOINT_API_KEY)
    try:
        obs_r = requests.get(obs_url)
    except RequestException as e:
        logger.error('Exception thrown when trying to read observation web service')
        return
    obs_status = obs_r.status_code
    if 200 <= obs_status < 300:
        try:
            obs_data = json.loads(obs_r.text)
        except JSONDecodeError as jde:
            logger.error('Unable to parse JSON describing weather observations passed from DataPoint')
            return
        if 'SiteRep' not in obs_data:
            logger.info('Found an obs_data with no SiteRep {}'.format(obs_data))
            return
        site_rep = obs_data['SiteRep']
        if 'DV' not in site_rep:
            logger.info('Found an SiteRep with no DV {}'.format(site_rep))
            return
        dv = (site_rep['DV'])
        if 'Location' not in dv:
            logger.info('Found a dv with no location {}'.format(dv))
            return
        periods = dv['Location']['Period']
        for period in periods:
            rep = (period['Rep'])
            for current_obs in rep:
                if 'Location' not in dv:
                    logger.error("'Location' is not in dv: {}".format(dv))
                    continue
                try:
                    time = "{} {}".format(
                        period['value'],
                        current_obs['$']
                    )
                    time = time.replace("Z", "")
                    times = time.split()
                    mins_since_midnight = int(times[1])
                    hours = mins_since_midnight // 60
                    minutes = mins_since_midnight % 60
                    time = "{} {}:{}".format(times[0], hours, minutes)
                    time = datetime.strptime(time, '%Y-%m-%d %H:%M')
                    time = pytz.utc.localize(time)

                    obs = WeatherObservation.objects.filter(
                        weather_location=location,
                        time_stamp=time
                    ).first()

                    if not obs:
                        obs = WeatherObservation()
                        obs.time_stamp = time
                        obs.weather_location = location
                        if 'D' in current_obs:
                            obs.wind_direction = current_obs['D']
                        if 'G' in current_obs:
                            obs.wind_gust = float(current_obs['G'])
                        if 'T' in current_obs:
                            obs.temperature = float(current_obs['T'])
                        if 'S' in current_obs:
                            obs.wind_speed = float(current_obs['S'])
                        if 'P' in current_obs:
                            obs.pressure = float(current_obs['P'])
                        if 'Pt' in current_obs:
                            obs.pressure_tendency = current_obs['Pt']
                        if 'Dp' in current_obs:
                            obs.dew_point = float(current_obs['Dp'])
                        if 'H' in current_obs:
                            obs.screen_relative_humidity = float(current_obs['H'])
                        if 'V' in current_obs:
                            obs.visibility = current_obs['V']
                        if 'D' in current_obs:
                            obs.wind_direction = current_obs['D']
                        if 'W' in current_obs:
                            obs.weather_type = int(current_obs['W'])
                        if 'U' in current_obs:
                            obs.max_uv = float(current_obs['U'])
                        if 'Pp' in current_obs:
                            obs.precipitation_probability = float(current_obs['Pp'])

                        obs.save()
                except Exception as e:
                    logger.exception('Exception thrown when saving observation', e)
                    return


def load_data():
    site_list_path = "val/wxobs/all/json/sitelist"
    # compatabilities_path = "val/wxobs/all/datatype/capabilities"
    obs_path = "val/wxobs/all/json/{}?res=hourly"

    site_list_url = "{}{}?key={}".format(settings.DATAPOINT_BASE_URL, site_list_path, settings.DATAPOINT_API_KEY)
    r = requests.get(site_list_url)
    status = r.status_code
    if 200 <= status < 300:
        try:
            data = json.loads(r.text)
        except JSONDecodeError as jde:
            logger.error('Unable to parse JSON describing weather locations passed from DataPoint')
            return
    else:
        logger.error("Denied access - HTTP response {}".format(status))
        return
    for location_current in data['Locations']['Location']:
        if BOTTOM_RIGHT[0] < float(location_current['latitude']) < TOP_LEFT[0] and \
                TOP_LEFT[1] < float(location_current['longitude']) < BOTTOM_RIGHT[1]:
            location = WeatherLocation.objects.filter(
                id=location_current['id']
            ).first()
            if location is None:
                location = WeatherLocation()

                location.id = location_current['id']
                location.name = location_current['name']
                location.region = location_current['region']
                location.unitaryAuthArea = location_current['unitaryAuthArea']
                location.latitude = location_current['latitude']
                location.longitude = location_current['longitude']
                location.elevation = location_current['elevation']
                location.save()
            store_observations(location, obs_path, location_current)
