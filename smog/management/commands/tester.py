from django.core.management.base import BaseCommand, CommandError
from ..test_metoffice_api import test_api


class Command(BaseCommand):
    def handle(self, *args, **options):
        test_api()
