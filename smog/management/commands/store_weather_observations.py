from django.core.management.base import BaseCommand

from ...business.datapoint.load_datapoint_data import load_data


class Command(BaseCommand):
    def handle(self, *args, **options):
        load_data()
