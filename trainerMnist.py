""" 
Nombre: Alejandro Tejada
Curso: ingeligencia artificial
Maestro: Samuel Chavez
Laboratorio #2
trainerMnist.py es el programa que entrena al modelos
"""


# --------------ZONA DE  IMPORTS
import mnist_reader  # ESTE mnist_reader lo que hace es que lee la data
from scipy.optimize import minimize
import numpy as np
from NeuralNetwork import *
import pickle

# ------------FIN ZONA LIBRERÍAS E IMPORTS

# ------------------ZONA DE DATOS, LECTURA Y CONSTANTES
X_train, y_train = mnist_reader.load_mnist('DatosTrainingYTest', kind='train')
X_test, y_test = mnist_reader.load_mnist('DatosTrainingYTest', kind='t10k')

# -------------------FIN ZONA DE DATOS, LECTURA Y CONSTANTES

# Constantes
'''
Factor por el que se divide para
que no baje tan rapido la sigmoide
'''
NORMALIZAR = 1000.0

# Cantidad de neuronas en la
# Capa oculta
NEURONAS_OCULTAS = 115

# Neuronas en la capa salida
NEURONAS_SALIDA = 10


# Proceso de datos
print("Proceso de datos...")
'''
Se leen los datos y se dividen por 
NORMALIZAR
'''
# X es la data de 28x28
# Info de cada pixel
X = X_train/NORMALIZAR

m, n = X.shape

# Y son las labels de cada imagen
y = y_train.reshape(m, 1)

Y = (y == np.array(range(10))).astype(int)

print("Se construye el modelo...")
# Se construye el modelo
theta_shapes = np.array([
    [NEURONAS_OCULTAS, n+1],
    [NEURONAS_SALIDA, NEURONAS_OCULTAS+1]
])

# Thetas iniciales
flat_thetas = flatten_list_of_arrays([
    np.random.rand(*theta_shape)/100
    for theta_shape in theta_shapes
])

print("Se entrena el modelo...")
# Optimizando
result = minimize(
    fun=cost_function,
    x0=flat_thetas,
    args=(theta_shapes, X, Y),
    method='L-BFGS-B',
    jac=back_propagation,
    options={'disp': True, 'maxiter': 900} #!Imporatante, número de vueltas que dará

)

print("Modelo entrenado! ")
modelo = open('modeloFinal', 'wb')
pickle.dump(result.x, modelo)

modelo.close()


# TEST

modeloCarga = open('modeloFinal', 'rb')
flat_thetas = pickle.load(modeloCarga)
modeloCarga.close()


X = X_test/NORMALIZAR

m, n = X.shape

# Y son las labels de cada imagen
y = y_test.reshape(m, 1)

Y = (y == np.array(range(10))).astype(int)


# Se construye el modelo
theta_shapes = np.array([
    [NEURONAS_OCULTAS, n+1],
    [NEURONAS_SALIDA, NEURONAS_OCULTAS+1]
])


resultado = feed_forward(
    inflate_matrixes(flat_thetas, theta_shapes),
    X
)

prediccion = np.argmax(resultado[-1], axis=1).reshape(m, 1)

correctos = ((prediccion == y)*1).sum()
incorrectos = len(y)-correctos

print("Correctas "+str(correctos))
print("Incorrectos "+str(incorrectos))
print("Exactitud "+str(correctos*100/float(len(y))))
