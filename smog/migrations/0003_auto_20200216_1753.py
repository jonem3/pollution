# Generated by Django 2.2.5 on 2020-02-16 17:53

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('smog', '0002_auto_20200215_1442'),
    ]

    operations = [
        migrations.CreateModel(
            name='PollutionLocation',
            fields=[
                ('site_code', models.CharField(max_length=255, primary_key=True, serialize=False)),
                ('latitude', models.FloatField(db_index=True)),
                ('longitude', models.FloatField(db_index=True)),
            ],
            options={
                'db_table': 'pollution_location',
            },
        ),
        migrations.AlterField(
            model_name='weatherlocation',
            name='latitude',
            field=models.FloatField(db_index=True),
        ),
        migrations.AlterField(
            model_name='weatherlocation',
            name='longitude',
            field=models.FloatField(db_index=True),
        ),
        migrations.AlterField(
            model_name='weatherobservation',
            name='time_stamp',
            field=models.DateTimeField(db_index=True),
        ),
        migrations.CreateModel(
            name='PollutionObservation',
            fields=[
                ('id', models.BigAutoField(primary_key=True, serialize=False)),
                ('time_stamp', models.DateTimeField(db_index=True)),
                ('species_code', models.CharField(max_length=255)),
                ('species_description', models.CharField(blank=True, max_length=255, null=True)),
                ('air_quality_index', models.IntegerField()),
                ('air_quality_band', models.CharField(max_length=255)),
                ('pollution_location', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='smog.PollutionLocation')),
            ],
            options={
                'db_table': 'pollution_observation',
                'unique_together': {('pollution_location', 'time_stamp')},
            },
        ),
    ]
