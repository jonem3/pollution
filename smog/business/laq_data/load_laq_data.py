from django.conf import settings
import time
import requests
import json
from datetime import datetime
import logging
import pytz
from json import JSONDecodeError
from requests.exceptions import RequestException

from smog.models import PollutionLocation

logger = logging.getLogger('quality_log')
BASE_URL = 'http://api.erg.kcl.ac.uk/AirQuality'
GROUPS_PATH = '/Information/Groups/Json'
LATEST_READING = '/Daily/MonitoringIndex/Latest/GroupName='
JSON = '/Json'
def store_readings(group_name):
    time.sleep(1)
    reading_url = '{}{}{}{}'.format(BASE_URL, LATEST_READING, group_name, JSON)
    print(reading_url)
    r = requests.get(reading_url)
    status = r.status_code
    if 200 <= status < 300:
        try:
            data = json.loads(r.text)
        except JSONDecodeError as jde:
            logger.error('Unable to parse JSON describing LAQ group data passed from LAQ')
    else:
        logger.error("Denied access - HTTP response {}".format(status))
    try:
        local_auths = data['DailyAirQualityIndex']['LocalAuthority']
        if type(local_auths) is not list:
            local_auths = [local_auths, ]
        for local_auth in local_auths:
            if 'Site' in local_auth:
                sites = local_auth['Site']
                if type(sites) is not list:
                    sites = [sites, ]
                for site in sites:
                    print(site['@SiteName'])
                    location = PollutionLocation.objects.filter(
                        site_code=site['@SiteCode']
                    ).first()
                    if location is None:
                        location = PollutionLocation()
                        location.site_code = site['@SiteCode']
                        location.site_name = site['@SiteName']
                        try:
                            location.latitude = float(site['@Latitude'])
                            location.longitude = float(site['@Longitude'])
                        except Exception as e:
                            print(site['@Latitude'], e)
                            continue

                        location.save()
    except Exception as e:
        logger.exception(e)
        exit(1)
def load_data():

    groups_url = "{}{}".format(BASE_URL, GROUPS_PATH)
    print(groups_url)
    r = requests.get(groups_url)
    status = r.status_code
    if 200 <= status < 300:
        try:
            data = json.loads(r.text)
        except JSONDecodeError as jde:
            logger.error('Unable to parse JSON describing LAQ locations passed from LAQ')
            return
    else:
            logger.error("Denied access - HTTP response {}".format(status))
    for group_current in data['Groups']['Group']:
        group_name = (group_current['@GroupName'])
        print(group_name)
        if group_name != 'All':
            store_readings(group_name)
