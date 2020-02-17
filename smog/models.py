from django.db import models


# Create your models here.

class WeatherLocation(models.Model):
    """
    {
        "elevation": "25.0",
        "id": "3772",
        "latitude": "51.479",
        "longitude": "-0.449",
        "name": "Heathrow",
        "region": "se",
        "unitaryAuthArea": "Greater London"
    }
    """
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    region = models.CharField(max_length=255)
    unitaryAuthArea = models.CharField(max_length=255)
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)
    elevation = models.FloatField()

    class Meta:
        db_table = 'weather_location'


class PollutionLocation(models.Model):
    site_code = models.CharField(primary_key=True, max_length=255)
    site_name = models.CharField(max_length=255, null=True, blank=True)
    latitude = models.FloatField(db_index=True)
    longitude = models.FloatField(db_index=True)

    class Meta:
        db_table = 'pollution_location'


class WeatherObservation(models.Model):
    id = models.BigAutoField(primary_key=True)
    weather_location = models.ForeignKey(WeatherLocation, models.CASCADE)
    time_stamp = models.DateTimeField(db_index=True)
    wind_gust = models.FloatField(blank=True, null=True)
    temperature = models.FloatField(blank=True, null=True)
    wind_speed = models.FloatField(blank=True, null=True)
    pressure = models.FloatField(blank=True, null=True)
    pressure_tendency = models.CharField(max_length=100, blank=True, null=True)
    dew_point = models.FloatField(blank=True, null=True)
    screen_relative_humidity = models.FloatField(blank=True, null=True)
    visibility = models.CharField(max_length=255, blank=True, null=True)
    wind_direction = models.CharField(max_length=255, blank=True, null=True)
    weather_type = models.IntegerField(blank=True, null=True)
    max_uv = models.FloatField(blank=True, null=True)
    precipitation_probability = models.FloatField(blank=True, null=True)

    class Meta:
        db_table = "weather_observation"
        unique_together = (('weather_location', 'time_stamp'),)

class PollutionObservation(models.Model):
    id = models.BigAutoField(primary_key=True)
    pollution_location = models.ForeignKey(PollutionLocation, models.CASCADE)
    time_stamp = models.DateTimeField(db_index=True)
    species_code = models.CharField(max_length=255, blank=False, null=False)
    species_description = models.CharField(max_length=255, blank=True, null=True)
    air_quality_index = models.IntegerField()
    air_quality_band = models.CharField(max_length=255)

    class Meta:
        db_table = 'pollution_observation'
        unique_together = (('pollution_location', 'time_stamp'),)
