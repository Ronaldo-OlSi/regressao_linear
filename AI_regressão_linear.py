from matplotlib import pyplot as plt1
import pandas as pd1
import numpy as np1
from math import sqrt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split

d_f = pd1.read_csv("Fuel_Consumption_Co2.csv")

print(d_f.head())
print(d_f.describe())

mts = d_f[['ENGINESIZE']]
co2 = d_f[['CO2EMISSIONS']]
print(mts.head())

motores_treino, mts_test, co2_tre, co2_tes = train_test_split(mts, co2, test_size= 0.2, random_state= 41)
print(type(motores_treino))

plt1.scatter(motores_treino, co2_tre, color='blue')
plt1.xlabel("Motor")
plt1.ylabel("Emissão de CO2")
plt1.show()

modelo = linear_model.LinearRegression()

modelo.fit(motores_treino, co2_tre)

print('(A) Intercepto: ', modelo.intercept_)
print('(B) Inclinação: ', modelo.coef_)

plt1.scatter(motores_treino, co2_tre, color='blue')
plt1.plot(motores_treino, modelo.coef_[0][0] * motores_treino + modelo.intercept_[0], '-r')
plt1.ylabel("Emissão de C02")
plt1.xlabel("Motores")
plt1.show()

predicoes_Co2 = modelo.predict(mts_test)

plt1.scatter(mts_test, co2_tes, color='blue')
plt1.plot(mts_test, modelo.coef_[0][0] * mts_test + modelo.intercept_[0], '-r')
plt1.ylabel("Emissão de C02")
plt1.xlabel("Motores")
plt1.show()

print("Soma dos Erros ao Quadrado (SSE): %.2f " % np1.sum((predicoes_Co2 - co2_tes) ** 2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_tes, predicoes_Co2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_tes, predicoes_Co2))
print("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_tes, predicoes_Co2)))
print("R2-score: %.2f" % r2_score(co2_tes, predicoes_Co2))