from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.cancer, name="cancer"),
    path('/predict', views.predict, name="predict"),
]