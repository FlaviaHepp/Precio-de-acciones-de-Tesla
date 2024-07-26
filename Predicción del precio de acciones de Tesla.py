""" Tesla , Inc., fundada en 2003 por los ingenieros Martin Eberhard y Marc Tarpenning , se ha convertido en líder en fabricación de vehículos 
eléctricos y soluciones de energía sostenible . El actual director ejecutivo de la empresa, Elon Musk, se unió poco después de su fundación y 
desempeñó un papel fundamental en su crecimiento e innovación. Tesla presentó su primer automóvil , el Roadster , en 2008, al que siguieron los 
exitosos Model S, Model X, Model 3 y Model Y. Más allá de los vehículos, Tesla se ha expandido hacia soluciones de almacenamiento de energía y 
productos solares, superando continuamente los límites. de tecnología y sostenibilidad. Con sede en Palo Alto, California, Tesla sigue a la 
vanguardia de la transición hacia las energías renovables y el transporte eléctrico .

Este conjunto de datos proporciona un registro completo de los cambios en el precio de las acciones de Tesla durante los últimos 14 años. Incluye 
columnas esenciales como la fecha , precio de apertura , precio más alto del día, precio más bajo del día, precio de cierre , precio de cierre 
ajustado y volumen de negociación .

Esta amplia información es invaluable para realizar análisis históricos , pronosticar el desempeño futuro de las acciones y comprender las 
tendencias del mercado a largo plazo relacionadas con las acciones de Tesla."""

import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os
import matplotlib.pyplot as plt
import plotly.express as px
from plotly import express
template = "plotly_dark"
import seaborn as sns


df = pd.read_csv("Tesla Dataset.csv")
df#mostrar los resultados

#verificar la información del precio de las acciones
df.info()
#de la salida, no faltan valores

#cambiar el formato a fecha y hora
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
print(df["Date"])

plt.style.use('dark_background')
plt.figure(figsize = (30,15))#ajustar el tamaño de la figura
plt.plot(df["Date"], df["Close"], color = "crimson", marker = "v")
plt.xticks(rotation = 45)#rotar el eje x 45 grados
plt.xlabel("Fecha\n")
plt.ylabel("Precio de cierre\n")
plt.title("Precios de cierre reales\n", fontsize = '16', fontweight = 'bold')
plt.grid()
plt.show()

##MODELO DE PREDICCIÓN
features = ["Open", "High", "Low", "Volume"]#selecciona los elementos
X = df[features]

y = df["Close"]#la meta

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()
model.fit(X_train, y_train)#entrenar el modelo

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print("Error medio cuadrado:", mse)

prediction_y = model.predict(X_test)#salida de predicción

dates =  df.iloc[X_test.index]["Date"]

prediction_df = pd.DataFrame({"Date": dates, "Predicted Closing Price": prediction_y})
prediction_df.sort_values("Date", inplace=True)

#dibuja el gráfico
plt.figure(figsize = (30,15))
plt.plot(df["Date"], df["Close"], label = "Precio de cierre real", color = "gold")
plt.plot(prediction_df["Date"], prediction_df["Predicted Closing Price"], label = "Precio de cierre previsto", color = "fuchsia")
plt.xlabel("Fecha\n")
plt.ylabel("Precio de cierre\n")
plt.title("Precios de cierre reales frente a previstos\n", fontsize = '16', fontweight = 'bold')
plt.legend()
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(df['Date'], df['Close'], color='cyan')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Fecha\n', fontsize=12)
ax.set_ylabel('Precio en USD\n', fontsize=12)
plt.title('Conjunto de datos de precios de acciones de Tesla\n', fontsize = '16', fontweight = 'bold')
plt.grid()
plt.show()

# Trama de barras
fig2, ax = plt.subplots(figsize=(20, 8))
ax.bar(df['Date'], df['Close'], color='green')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Fecha\n', fontsize=12)
ax.set_ylabel('Precio en USD\n', fontsize=12)
plt.title('Conjunto de datos de precios de acciones de Tesla\n', fontsize = '16', fontweight = 'bold')
plt.grid()
plt.show()


df.hist(bins = 20, figsize = (20,20), color = 'g')
plt.show()

df.columns.to_list()

for column in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
    express.histogram(data_frame=df, x=column, template = template).show()
    
numeric_cols = df.select_dtypes(include=np.number).columns  

plt.figure(figsize=(15, 15))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='spring')
plt.title('Matriz de correlación\n', fontsize = '16', fontweight = 'bold')
plt.show()

features = ["Open", "High", "Low", "Volume"]
X = df[features]

y = df["Close"]
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

print("Error medio cuadrado:", mse)

prediction_y = model.predict(X_test)#salida de predicción

dates =  df.iloc[X_test.index]["Date"]

prediction_df = pd.DataFrame({"Date": dates, "Predicted Closing Price": prediction_y})
prediction_df.sort_values("Date", inplace=True)

