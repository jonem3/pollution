from django.core.management.base import BaseCommand

from smog.business.tensorflow_building.building_tensorflow_model import build_model


class Command(BaseCommand):
    def handle(self, *args, **options):
        build_model()