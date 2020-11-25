from django.urls import path
from .view import pollution, about
from .views.home import home_page
urlpatterns = [
    path('', home_page, name='main'),
    path('pollution.json', pollution, name='pollution'),
    path('about/', about, name='about')

]