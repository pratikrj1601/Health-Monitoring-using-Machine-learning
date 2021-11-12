from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.liver, name="liver"),
    path('/predict', views.predict, name="predict"),
]