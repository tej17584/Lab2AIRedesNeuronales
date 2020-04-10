""" 
Nombre: Alejandro Tejada
Curso: ingeligencia artificial
Maestro: Samuel Chavez
Laboratorio #2
trainerMnist.py es el programa que entrena al modelo

"""


#--------------ZONA DE  IMPORTS
import mnist_reader  #ESTE mnist_reader lo que hace es que lee la data 
import numpy as np
#------------FIN ZONA LIBRERÍAS E IMPORTS


X_train, y_train = mnist_reader.load_mnist('DatosTrainingYTest', kind='train')
X_test, y_test = mnist_reader.load_mnist('DatosTrainingYTest', kind='t10k')