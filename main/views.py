import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from django.shortcuts import render


# Create your views here.

def home(request):
    return render(request, "home.html")


def predict(request):
    return render(request, "predict.html")


def index(request):
    return render(request, "index.html")


def result(request):
    data = pd.read_csv(r'dataset/USA_Housing.csv')
    new_data = data.drop(['Address'], axis=1)
    X = new_data.drop('Price', axis=1)
    Y = new_data['Price']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=20)
    model = LinearRegression()
    model.fit(X_train, Y_train)

    avg_area_income = float(request.GET['n1'])
    avg_area_house_age = float(request.GET['n2'])
    avg_area_num_rooms = float(request.GET['n3'])
    avg_area_num_bedrooms = float(request.GET['n4'])
    avg_area_population = float(request.GET['n5'])

    prediction = model.predict(
        np.array([avg_area_income, avg_area_house_age, avg_area_num_rooms, avg_area_num_bedrooms,
                  avg_area_population]).reshape(1, -1))
    prediction = round(prediction[0])

    price = f"The predicted price is {prediction}"

    return render(request, "predict.html", {"result2": price})
