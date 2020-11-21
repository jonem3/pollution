from django.http.response import HttpResponse
from django.shortcuts import render
from smog.models import WeatherLocation

def home_page(request):
    locations = WeatherLocation.objects.order_by('name').all()
    return render(request, 'main.html',
                  {'locations': locations})