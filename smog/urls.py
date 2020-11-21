from django.urls import path
from .view import pollution
from .views.home import home_page
urlpatterns = [
    path('', home_page, name='main'),
    path('pollution', pollution, name='pollution')
]