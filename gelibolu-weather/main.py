import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import statsmodels.api as sm

data = pd.read_csv("weather.csv",sep=";")
data.drop("Sunshine Duration",axis=1, inplace=True)
data.drop("Shortwave Radiation",axis=1, inplace=True)
data.drop("DateTime",axis=1, inplace=True)
data.drop("Mean Sea Level Pressure",axis=1, inplace=True)
data.drop("Wind Direction",axis=1, inplace=True)
data.columns = ["Sıcaklık","Nem","Toprak Sıcaklık","Toprak Nem","Rüzgar Hız"]


x = data.drop("Sıcaklık", axis=1).values
y = data["Sıcaklık"].values

x_tr, x_te, y_tr, y_te = train_test_split(x,y, test_size=0.1)
model = LinearRegression()
model.fit(x_tr, y_tr)

y_p = model.predict(x_te)
print(r2_score(y_te,y_p))
print(model.score(x_te,y_te))


o = sm.OLS(y_tr,x_tr).fit()
print(o.summary())


plt.figure(figsize=(10,10))
plt.plot(data["Sıcaklık"].sort_values(),data["Nem"].sort_values())
plt.show()

plt.figure(figsize=(10,10))
plt.plot(data["Sıcaklık"].sort_values(),data["Toprak Sıcaklık"].sort_values())
plt.show()

plt.figure(figsize=(10,10))
plt.plot(data["Sıcaklık"].sort_values(),data["Toprak Nem"].sort_values())
plt.show()

plt.figure(figsize=(10,10))
plt.plot(data["Sıcaklık"].sort_values(),data["Rüzgar Hız"].sort_values())
plt.show()


plt.figure(figsize=(10,10))
plt.subplot(221).plot(data["Sıcaklık"].sort_values(),data["Nem"].sort_values())
plt.subplot(222).plot(data["Sıcaklık"].sort_values(),data["Toprak Sıcaklık"].sort_values())
plt.subplot(223).plot(data["Sıcaklık"].sort_values(),data["Toprak Nem"].sort_values())
plt.subplot(224).plot(data["Sıcaklık"].sort_values(),data["Rüzgar Hız"].sort_values())
plt.show()