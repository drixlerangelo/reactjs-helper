from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home.index'),
    path('ask/', views.ask, name='home.ask'),
]
