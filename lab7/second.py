from typing import Tuple, Callable
import numpy as np


def MRS(start_x, end_x, points_num,
        bound_cond_A: Tuple,
        bound_cond_B: Tuple, ):
    x_values = np.linspace(start_x, end_x, points_num)
    h = x_values[1] - x_values[0]

    vector = np.zeros(points_num + 1)
    matrix = np.zeros(shape=(points_num + 1, points_num + 1))

    if bound_cond_A[0] == 0:
        matrix[0, 0] = 1
        vector[0] = bound_cond_A[1]

    for i in range(1, points_num):
        for j in range(points_num + 1):
            if i == j:
                matrix[i, j] = -2
            elif i == j + 1 and j + 1 < points_num + 1:
                matrix[i, j] = 1
            elif i == j - 1 and j - 1 >= 0:
                matrix[i, j] = 1

    for i in range(1, points_num):
        vector[i] = h ** 2 * x_values[i]

    if bound_cond_B[0] == 0:
        matrix[-1] = bound_cond_B[1]

    # bound = (y-3, y-1, vector)
    elif bound_cond_B[0] == 1:
        matrix[points_num, -1] = bound_cond_B[1]
        matrix[points_num, -3] = bound_cond_B[2]
        vector[-1] = bound_cond_B[3] * 2 * h

    print(x_values)
    print(vector)
    print(matrix)
    vec = np.linalg.solve(matrix, vector)
    return vec[:-1]


def MRS_zad(start_x, end_x, points_num,
        bound_cond_A: Tuple,
        bound_cond_B: Tuple, fun=lambda x: x):
    x_values = np.linspace(start_x, end_x, points_num)
    h = x_values[1] - x_values[0]

    vector = np.zeros(points_num + 1)
    matrix = np.zeros(shape=(points_num + 1, points_num + 1))

    if bound_cond_A[0] == 0:
        matrix[0, 0] = 1
        vector[0] = bound_cond_A[1]

    for i in range(1, points_num):
        for j in range(points_num + 1):
            if i == j:
                matrix[i, j] = -2 + 25
            elif i == j + 1 and j + 1 < points_num + 1:
                matrix[i, j] = 1
            elif i == j - 1 and j - 1 >= 0:
                matrix[i, j] = 1

    for i in range(1, points_num):
        vector[i] = h ** 2 * fun(x_values[i])

    if bound_cond_B[0] == 0:
        matrix[-1] = bound_cond_B[1]

    # bound = (y-3, y-1, vector)
    elif bound_cond_B[0] == 1:
        matrix[points_num, -1] = bound_cond_B[1]
        matrix[points_num, -3] = bound_cond_B[2]
        vector[-1] = bound_cond_B[3] * 2 * h

    # print(x_values)
    # print(vector)
    # print(matrix)
    vec = np.linalg.solve(matrix, vector)
    return vec[:-1]


def zad():
    m = 5

    def fun_x(x):
        return np.cos(m * x) - x * np.sin(m * x)

    def fun_right(x):
        return -10 * np.cos(m * x)

    x0 = 0
    y0 = 1
    xk = (np.pi * 2 + 2) / m
    yk = fun_x(xk)

    points = 10

    ret = MRS_zad(x0, xk, points, (0, y0), (0, yk), fun_right)

    x = np.linspace(x0, xk, 1000)
    y = list(map(fun_right, x))



if __name__ == "__main__":

    # test function result: [2, 1.5, 1.25, 1., 1.25]
    # r = MRS(0, 2, 5, (0, 2), (1, 1, -1, 1))
    # print(r)

    zad()
