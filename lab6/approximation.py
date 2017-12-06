import lab1.matrix_gauss as mg
import matplotlib.pyplot as plt
import numpy as np
from typing import List


def approximate(polynominal_degree: int, x_val: List[float], y_val: List[float]):
    data_size = len(x_val)
    assert data_size == len(y_val)
    assert polynominal_degree >= 0
    polynominal_degree += 1

    vec_b = []
    matrix_A = []

    for k in range(0, polynominal_degree):
        vec_b_sum = 0
        for j in range(data_size):
            x, y = x_val[j], y_val[j]
            vec_b_sum += y * x ** k
        vec_b.append([vec_b_sum])

        row = []
        for i in range(polynominal_degree):
            fun_sum = 0
            for j in range(data_size):
                fun_sum += x_val[j] ** (i + k)
            row.append(fun_sum)
        matrix_A.append(row)

    coef = mg.gauss_elimination(matrix_A, vec_b)
    coef = [row[0] for row in coef]

    def inner(x_arg):
        total_val = 0
        for i, a in enumerate(coef):
            total_val += a * x_arg ** i
        return total_val

    return inner


def f(x):
    return 3 + 4 * x + 2 * np.e ** x


if __name__ == "__main__":
    left = -10
    right = 10
    points_num = 20

    x = list(np.linspace(left, right, points_num))
    y = list(map(f, x))

    approx = approximate(4, x, y)

    xx = np.linspace(left, right, 1000)
    yy = list(map(f, xx))
    yy_aprox = list(map(approx, xx))

    plt.plot(xx, yy, label="f(x)=...")
    plt.plot(xx, xx * 0, label="f(x)=0")
    plt.scatter(x, y)
    plt.plot(xx, yy_aprox, label="f(x) aprox")

    plt.ylabel("y=f(x)")
    plt.xlabel("x watosci")
    plt.legend()
    plt.show()
