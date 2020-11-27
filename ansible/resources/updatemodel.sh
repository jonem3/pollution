#!/usr/bin/env bash

service apache2 stop

source /home/ubuntu/pollution/pollutionenv/bin/activate
python /home/ubuntu/pollution/manage.py alternative
python /home/ubuntu/pollution/manage.py collectstatic --noinput

service apache2 start