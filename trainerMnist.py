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

# ----------------------------------ZONA CONSTANTES*--------------
'''
Factor para que la sigmoide no baje
'''
FactorNormal = 1000.0  # normalizacion
CNeuronas = 136  # neuronas ocultas
CNeuronasSalida = 10  # salida
# ----------------------FIN ZONA CONSTANTES
# Proceso de datos
print("Proceso de datos...")
'''
Se leen los datos y se dividen por 
FactorNormal
'''
# X es la data de 28x28
# Info de cada pixel
X = X_train/FactorNormal
m, n = X.shape  # se usa el shape
# labels
y = y_train.reshape(m, 1)
Y = (y == np.array(range(10))).astype(int)  # arrange

print("Construccion del modelo.")
# la construccion del modelo implica shapes de thetas
theta_shapes = np.array([
    [CNeuronas, n+1],
    [CNeuronasSalida, CNeuronas+1]
])
"""
! Construccion de thetas iniciales, deben ser random
"""
flat_thetas = flatten_list_of_arrays([
    np.random.rand(*shape)/100
    for shape in theta_shapes
])

# !Importante, este es el idea, la optimización debe ser así por sugerencia dle maestro
resultadoMinimize = minimize(
    fun=cost_function,
    x0=flat_thetas,
    args=(theta_shapes, X, Y),
    method='L-BFGS-B',
    jac=back_propagation,
    # !Imporatante, número de vueltas que dará el moelo
    options={'disp': True, 'maxiter': 900}

)

print("El modelo fue entrando")
# Usando picke para poder hacer la
modeloEntrenado = open('modeloThetas', 'wb')
pickle.dump(resultadoMinimize.x, modeloEntrenado)
# ceramos con pickle
modeloEntrenado.close()
