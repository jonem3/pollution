from django.core.management.base import BaseCommand

from smog.business.tensorflow_deployment.tensorflow_deploy import get_data


class Command(BaseCommand):
    def handle(self, *args, **options):
        get_data()