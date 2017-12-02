import numpy as np
from numpy import arange, array, linspace, ones, zeros
from scipy.linalg import solve_banded
from typing import List
import matplotlib.pyplot as plt
from lab1.band_matrix import thomas_algorithm
from itertools import zip_longest


def coef2(x_values, y_values, alpha=0, beta=0):
    n = len(x_values)
    assert n == len(y_values)

    top, mid, bot = [0] * n, [0] * n, [0] * n
    n -= 1
    h = [x_values[i + 1] - x_values[i] for i in range(n)]

    # first row values
    vector_b = [0] * (n + 1)
    top[0] = 0
    mid[0] = 1
    bot[0] = None

    # prepare vector b and tridiagonal matrix A
    for i in range(1, n):
        part1 = 3 * (y_values[i + 1] - y_values[i]) / h[i]
        part2 = 3 * (y_values[i] - y_values[i - 1]) / h[i - 1]
        vector_b[i] = part1 - part2

        top[i] = h[i]
        mid[i] = 2 * (h[i - 1] + h[i])
        bot[i] = h[i - 1]

    # last row values
    top[n] = None
    mid[n] = 1
    bot[n] = 0

    # bound conditions
    vector_b[0] = alpha
    vector_b[n] = beta

    # solve by thomas algorithm A*x=b
    c = thomas_algorithm((top, mid, bot), [[i] for i in vector_b])
    c = [i[0] for i in c]

    a = y_values[:]
    b, d = [0] * n, [0] * n
    for i in range(n - 1, -1, -1):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (h[i] * 3)

    return list(zip_longest(a, b, c[:-1], d, x_values))


def spline_to_x_y(coef_and_x, num_of_points):
    n = len(coef_and_x) - 1
    spline_points = num_of_points / n
    if spline_points < 3:
        spline_points = 3

    x, y = [], []
    for i in range(n):
        a, b, c, d, x0 = coef_and_x[i]
        x1 = coef_and_x[i + 1][4]

        for xi in np.linspace(x0, x1, spline_points):
            x.append(xi)
            h = xi - x0
            y.append(a + b * h + c * h ** 2 + d * h ** 3)

    return x, y


def plot_function(fun=lambda x: x ** 2):
    left = -1
    right = 1
    n = 10
    x = np.linspace(left, right, n)
    y = list(map(fun, x))

    num_of_points = 100

    h = right - left
    h /= n
    # a, b, c, d = coef(h, x, y)
    coef_and_x = coef2(x, y)

    xx, yy = spline_to_x_y(coef_and_x, num_of_points)

    plt.subplot(111)
    plt.plot(xx, yy, label="interpolacja")
    plt.scatter(x, y)
    org_x = np.linspace(left, right, num_of_points)
    org_y = list(map(fun, org_x))
    plt.plot(org_x, org_y, label="org")
    plt.plot(x, x * 0, label="f(x)=0")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_function()
