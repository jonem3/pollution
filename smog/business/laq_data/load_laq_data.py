from django.conf import settings
import time
import requests
import json
from datetime import datetime
import logging
from json import JSONDecodeError
from requests.exceptions import RequestException
import pytz
from smog.models import PollutionLocation, PollutionObservation

logger = logging.getLogger('quality_log')
BASE_URL = 'http://api.erg.kcl.ac.uk/AirQuality'
GROUPS_PATH = '/Information/Groups/Json'
LATEST_READING = '/Daily/MonitoringIndex/Latest/GroupName='
JSON = '/Json'


def store_readings(group_name):
    time.sleep(1)
    reading_url = '{}{}{}{}'.format(BASE_URL, LATEST_READING, group_name, JSON)
    logger.debug(reading_url)
    try:
        r = requests.get(reading_url)
    except RequestException as e:
        logger.error('Exception thrown when trying to read observation web service')
        return
    status = r.status_code
    if 200 <= status < 300:
        try:
            data = json.loads(r.text)
        except JSONDecodeError as jde:
            logger.error('Unable to parse JSON describing LAQ group data passed from LAQ')
            return
    else:
        logger.error("Denied access - HTTP response {}".format(status))
        return
    try:
        if 'DailyAirQualityIndex' not in data:
            logger.error("Daily Air Quality Index not present, ignoring location")
            return
        dailyairqualityindex = data['DailyAirQualityIndex']
        if 'LocalAuthority' not in dailyairqualityindex:
            logger.error("Local Authority not present, ignoring location")
            return
        local_auths = dailyairqualityindex['LocalAuthority']
        if type(local_auths) is not list:
            local_auths = [local_auths, ]
        for local_auth in local_auths:
            if 'Site' not in local_auth:
                logger.debug("Found local_auth without a site")
                continue
            sites = local_auth['Site']
            if type(sites) is not list:
                sites = [sites, ]
            for site in sites:
                if '@SiteName' not in site or '@SiteCode' not in site:
                    logger.error("Site Name and Site Code data not present, ignoring site")
                    return
                logger.debug(site['@SiteName'])
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
                        logger.debug("Latitude and Longitude not present", e)
                        continue
                    location.save()

                if '@BulletinDate' not in site:
                    logger.debug("No Bulletin Date")
                    continue
                times = site['@BulletinDate']
                times = datetime.strptime(times, "%Y-%m-%d %H:%M:%S")
                times = pytz.utc.localize(times)
                if 'Species' not in site:
                    continue
                species = site['Species']
                if type(species) is not list:
                    species = [species, ]
                for classification in species:
                    observation = PollutionObservation.objects.filter(
                        time_stamp=times,
                        species_code=classification['@SpeciesCode'],
                        pollution_location=location
                    ).first()
                    if observation is None:
                        observation = PollutionObservation()
                        observation.pollution_location = location
                        observation.time_stamp = times
                        if '@SpeciesCode' in classification:
                            observation.species_code = classification['@SpeciesCode']
                        if '@SpeciesDescription' in classification:
                            observation.species_description = classification['@SpeciesDescription']
                        observation.air_quality_index = classification['@AirQualityIndex']
                        observation.air_quality_band = classification['@AirQualityBand']
                        observation.save()
    except Exception as e:
        logger.exception(e)
        exit(1)


def load_data():
    groups_url = "{}{}".format(BASE_URL, GROUPS_PATH)
    logger.debug(groups_url)
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
        return
    for group_current in data['Groups']['Group']:
        group_name = (group_current['@GroupName'])
        logger.debug(group_name.upper())
        if group_name != 'All':
            store_readings(group_name)