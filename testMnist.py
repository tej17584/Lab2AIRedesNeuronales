""" 
Nombre: Alejandro Tejada
Curso: ingeligencia artificial
Maestro: Samuel Chavez
Laboratorio #2
testMnis.py es el programa que hace el test
"""


# --------------ZONA DE  IMPORTS
import mnist_reader  # ESTE mnist_reader lo que hace es que lee la data
from scipy.optimize import minimize
import numpy as np
from NeuralNetwork import *
import pickle
import matplotlib.pyplot as pd
import mnist_reader
import pandas as plt
# ------------FIN ZONA LIBRERÍAS E IMPORTS

# ----------------------------------ZONA CONSTANTES*--------------
'''
Factor para que la sigmoide no baje
'''
FactorNormal = 1000.0  # normalizacion
CNeuronas = 136  # neuronas ocultas
CNeuronasSalida = 10  # salida
# ----------------------FIN ZONA CONSTANTES
# Lectura de datos
print("Lectura de los datos de la ropa")
X_test, y_test = mnist_reader.load_mnist('DatosTrainingYTest', kind='t10k')

#! Cargamos el modelo
modeloCarga = open('modeloThetas', 'rb')
flat_thetas = pickle.load(modeloCarga)
modeloCarga.close()
X = X_test/FactorNormal
m, n = X.shape

# Labels d elas imagnes
y = y_test.reshape(m, 1)
Y = (y == np.array(range(10))).astype(int)
# construccion del modelo
theta_shapes = np.array([
    [CNeuronas, n+1],
    [CNeuronasSalida, CNeuronas+1]
])
# resultado
resultado = feed_forward(
    inflate_matrixes(flat_thetas, theta_shapes),
    X
)
# Prediccion de valores
#! esta prediccion se toma la ultiam
prediccion = np.argmax(resultado[-1], axis=1).reshape(m, 1)

correctos = ((prediccion == y)*1).sum()
incorrectos = len(y)-correctos

print("Valores Correctos:.... "+str(correctos))
print("Prediccions incorrectas: ......"+str(incorrectos))
print("Exactitud para este modelo: .........." +
      str(correctos*100/float(len(y))))

prediccionMal = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(len(y)):
    if(not(y[i] == prediccion[i])):
        prediccionMal[int(y[i])] += 1
# ploteamos los malos
pd.title('Cantidad de errores por categoría')
pd.xlabel('Categoría/Label')
pd.ylabel('Errores')
pd.bar(range(len(prediccionMal)), prediccionMal, color=(1, 0.8, 0.2, 1))
pd.show()
