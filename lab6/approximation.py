import lab1.matrix_gauss as mg
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import random


def approximate(polynominal_degree: int, x_val: List[float],
                y_val: List[float]):
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


def norm(y_original, y_approximated):
    norm_max = 0
    norm_2 = 0
    for y1, y2 in zip(y_original, y_approximated):
        norm_max = max(norm_max, abs(y1 - y2))
        norm_2 += (y1 - y2) * (y1 - y2)
    return norm_max, norm_2


def f(x):
    return 3 + 4 * x + 2 * np.e ** x


def fun(x):
    return x * np.sin(0.8 * np.pi / x)


if __name__ == "__main__":
    left = 0.1
    right = 0.8
    points_num = 40

    # left = -10
    # right = 6
    # def ff(x):
    #     return np.e ** (-3 *np.sin(x))
    # fun = ff
    # points_num = 3

    # x = [left + random.random() * (right - left)
    #      for _ in range(points_num - 2)]
    # x = [left] + x + [right]

    points_num = 20
    print("liczba punktów{}".format(points_num))
    for i in range(1, 95):
        # points_num = i
        degree = i

        x = list(np.linspace(left, right, points_num))
        y = list(map(fun, x))

        approx = approximate(degree, x, y)

        xx = np.linspace(left, right, 1000)
        yy = list(map(fun, xx))
        yy_aprox = list(map(approx, xx))

        # print("liczba punktów: {}"
        #       " dokladność w metryce maksimum: {:.3f},"
        #       " średniokwadratowa: {:.3f}"
        print("{} {:2.4f} {:2.4f}".format(i, *norm(yy, yy_aprox)))

        if i == 10:
            plt.title("Liczba węzłów: 10, stopien wielomianu: 7")
            plt.plot(xx, yy, label="f(x)=x*sin(0.8*pi/x)")
            plt.plot(xx, xx * 0, label="f(x)=0")
            plt.scatter(x, y)
            plt.plot(xx, yy_aprox, label="f(x) aproymowana")

            plt.ylabel("f(x)")
            plt.xlabel("x")
            plt.legend()
            plt.show()
