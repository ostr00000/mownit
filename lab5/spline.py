import numpy as np
from typing import List
import matplotlib.pyplot as plt
from lab1.band_matrix import thomas_algorithm
from itertools import zip_longest


def derivative(func, eps=1e-6):
    def inner(x, n=1):
        if n == 1:
            return (func(x + eps) - func(x - eps)) / eps / 2
        return (inner(x + eps, n - 1) - inner(x - eps, n - 1)) / eps / 2

    return inner


def fun(x):
    return x * np.sin(0.8 * np.pi / x)


fun_der = derivative(fun)
left, right = 0.1, 0.8


def coef(x_val: List[float], y_val: List[float],
         bound_condition='natural', alpha=0, beta=0):
    n = len(x_val)
    assert n == len(y_val)

    top, mid, bot = [0] * n, [0] * n, [0] * n
    vector_b = [0] * n
    n -= 1
    h = [x_val[i + 1] - x_val[i] for i in range(n)]

    if bound_condition == 'natural':
        # first row values
        top[0] = 0
        mid[0] = 1
        bot[0] = None

        # last row values
        top[n] = None
        mid[n] = 1
        bot[n] = 0

        # vector_b bound conditions
        vector_b[0] = alpha
        vector_b[n] = beta

    elif bound_condition == 'clamped':
        top[0] = h[0]
        mid[0] = 2 * h[0]
        bot[0] = None

        top[n] = None
        mid[n] = 2 * h[-1]
        bot[n] = h[-1]

        vector_b[0] = -0.5 * alpha * h[0] ** 2 + 3 * (y_val[1] - y_val[0])
        vector_b[n] = 0.5 * beta * h[-1] ** 2 + 3 * (y_val[-1] - y_val[-2])

    else:
        raise TypeError("unknown boundary condition")

    # prepare vector b and tridiagonal matrix A
    for i in range(1, n):
        part1 = 3 * (y_val[i + 1] - y_val[i]) / h[i]
        part2 = 3 * (y_val[i] - y_val[i - 1]) / h[i - 1]
        vector_b[i] = part1 - part2

        top[i] = h[i]
        mid[i] = 2 * (h[i - 1] + h[i])
        bot[i] = h[i - 1]

    # solve by thomas algorithm A*x=b
    c = thomas_algorithm((top, mid, bot), [[i] for i in vector_b])
    c = [i[0] for i in c]

    a = y_val[:]
    b, d = [0] * n, [0] * n
    for i in range(n - 1, -1, -1):
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (h[i] * 3)

    return list(zip_longest(a, b, c[:-1], d, x_val))


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


def plot_function(fun=lambda x: x ** 2, left=-1., right=1., num=10):
    x = np.linspace(left, right, num)
    y = list(map(fun, x))
    coef_and_x = coef(list(x), y,
                      alpha=fun_der(left, 2),
                      beta=fun_der(right, 2))

    plot_points = 1000
    plt.subplot(111)

    # plot interpolated function
    xx, yy = spline_to_x_y(coef_and_x, plot_points)
    plt.plot(xx, yy, label="interpolacja f(x)=x*sin(0.8*pi/x)")

    # plot original function
    org_x = np.linspace(left, right, plot_points)
    org_y = list(map(fun, org_x))
    plt.plot(org_x, org_y, label="f(x)=x*sin(0.8*pi/x)")

    # plot interpolated points
    plt.scatter(x, y)

    # plot f(x) = 0
    plt.plot(x, x * 0, label="f(x)=0")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    fun = lambda x: x ** 2
    plot_function(fun, left, right, num=10)
