# Generated by Django 2.2.5 on 2020-01-05 14:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='WeatherLocation',
            fields=[
                ('id', models.IntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('region', models.CharField(max_length=255)),
                ('unitaryAuthArea', models.CharField(max_length=255)),
                ('latitude', models.FloatField()),
                ('longitude', models.FloatField()),
                ('elevation', models.FloatField()),
            ],
            options={
                'db_table': 'weather_location',
            },
        ),
        migrations.CreateModel(
            name='WeatherObservation',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('time_stamp', models.DateTimeField()),
                ('wind_gust', models.FloatField(blank=True, null=True)),
                ('temperature', models.FloatField(blank=True, null=True)),
                ('wind_speed', models.FloatField(blank=True, null=True)),
                ('pressure', models.FloatField(blank=True, null=True)),
                ('pressure_tendency', models.FloatField(blank=True, null=True)),
                ('dew_point', models.FloatField(blank=True, null=True)),
                ('screen_relative_humidity', models.FloatField(blank=True, null=True)),
                ('visibility', models.CharField(blank=True, max_length=255, null=True)),
                ('wind_direction', models.CharField(blank=True, max_length=255, null=True)),
                ('weather_type', models.IntegerField(blank=True, null=True)),
                ('max_uv', models.FloatField(blank=True, null=True)),
                ('precipitation_probability', models.FloatField(blank=True, null=True)),
                ('weather_location', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='smog.WeatherLocation')),
            ],
            options={
                'db_table': 'weather_observation',
                'unique_together': {('weather_location', 'time_stamp')},
            },
        ),
    ]
