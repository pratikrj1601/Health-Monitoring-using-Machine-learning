from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.heart_disease, name="heart_disease"),
    path('/predict', views.predict, name="predict")
]