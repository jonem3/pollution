from django.core.management.base import BaseCommand

from ...business.laq_data.load_laq_data import load_data


class Command(BaseCommand):
    def handle(self, *args, **options):
        load_data()
