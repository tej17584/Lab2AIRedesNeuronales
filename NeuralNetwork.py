""" 
Nombre: Alejandro Tejada
Curso: ingeligencia artificial
Maestro: Samuel Chavez
Laboratorio #2
NeuraNetwork.py es el programa que contiene las definiciones de feedfordward, etc...
"""
# --------------ZONA DE  IMPORTS
import numpy as np
from functools import reduce
# ------------FIN ZONA LIBRERÍAS E IMPORTS


""" 
devuelve matrices adecuadas
!thetas_flat son los valores
!shapes son los  pesos
"""


def inflate_matrixes(thetas_flat, shapes):
    capas = len(shapes) + 1
    sizes = [shape[0] * shape[1] for shape in shapes]
    steps = np.zeros(capas, dtype=int)

    # se hace el for
    for i in range(capas - 1):
        steps[i + 1] = steps[i] + sizes[i]
    # se retorna inflada la matriz
    return [
        thetas_flat[steps[i]: steps[i + 1]].reshape(*shapes[i])
        for i in range(capas - 1)
    ]


""" 
Función Sigmoide, recibe como parámetro una matriz
* sfdfd
! mat es la matriz de sigmoide que recibe
"""


def sigmoid(mat):
    a = [(1 / (1 + np.exp(-x))) for x in mat]
    return np.asarray(a).reshape(mat.shape)


""" 
Feed Forward Encuentra las funciones de activacion de cada neuona
! mat es la matriz de sigmoide que recibe
"""


def feed_forward(array_thetas, X):
    mat_list = [np.asarray(X)]  # PASO 2.1
    # For (2.2):
    for i in range(len(array_thetas)):
        mat_list.append(
            sigmoid(  # aplicar sigmoide
                np.matmul(  # Multiplicación de matrices (matricial)
                    np.hstack((
                        #! IMPORTANTE EL BIAS
                        np.ones(len(X)).reshape(len(X), 1),
                        mat_list[i]
                    )), array_thetas[i].T
                )
            )
        )
    return mat_list


""" 
cost_function se usa para obtener la prediccion
! a es el retorno del feed_forward
"""


def cost_function(thetas_flat, shapes, X, Y):
    # a será nuesto retornno
    a = feed_forward(inflate_matrixes(thetas_flat, shapes), X)
    # retornamos
    return -(Y * np.log(a[-1]) + (1 - Y) * np.log(1 - a[-1])).sum() / len(X)


""" 
cost_function se usa para obtener la prediccion
! X son las entradas de la red
! Y es el valor real de la prediccion
! los pesos son las thtetas
"""


def back_propagation(thetas_flat, shapes, X, Y):
    m, capas = len(X), len(shapes) + 1
    thetas = inflate_matrixes(thetas_flat, shapes)
    a = feed_forward(thetas, X)  # PASO 2.2

    deltas = [*range(capas - 1), a[-1] - Y]  # PASO 2.4
    for i in range(capas - 2, 0, -1):  # este es el for del 2.4
        deltas[i] = (deltas[i + 1] @ np.delete((thetas[i]), 0, 1)
                     ) * (a[i] * (1 - a[i]))

    deltasFinal = []
    for i in range(capas - 1):  # paso 2.5
        deltasFinal.append(
            (
                deltas[i + 1].T @
                np.hstack((
                    np.ones(len(a[i])).reshape(len(a[i]), 1),
                    a[i]
                ))
            ) / m
        )

    deltasFinal = np.asarray(deltasFinal)

#! retorna lista de arreglos
    return flatten_list_of_arrays(
        deltasFinal
    )


labelsCloths = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


#! lista de métodos reduce
def flatten_list_of_arrays(list_of_arrays): return reduce(
    lambda acc, v: np.array([*acc.flatten(), *v.flatten()]),
    list_of_arrays
)
