from django.urls import path
from .view import pollution, about
from .views.home import home_page
from django.views.decorators.cache import cache_page
urlpatterns = [
    path('', cache_page(60 * 30)(home_page), name='main'),
    path('pollution.json', pollution, name='pollution'),
    path('about/', cache_page(60 * 30)(about), name='about')

]