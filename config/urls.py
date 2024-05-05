from django.contrib import admin
from django.urls import path

from main.views import home, predict, result, index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home),
    path('index/', index),
    path('predict/', predict),
    path('predict/result/result/', result)
]
