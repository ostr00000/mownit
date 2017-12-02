import numpy as np
from typing import List
import matplotlib.pyplot as plt
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


def coef(x_val: List[float], y_val: List[float], alpha=0, beta=0):
    n = len(x_val)
    assert n == len(y_val)

    # | mid[0] top[0] ...                         | | tmp[0] | | vector_b[0] |
    # | bot[1] mid[1] top[1] ...                  | | tmp[1] | | vector_b[1] |
    # | ...    bot[2] mid[2] top[2] ...           | | tmp[2] | | vector_b[2] |
    # |                 ...                       |x|   ...  |=|     ...     |
    # | ...             bot[n-1] mid[n-1] top[n-1]| |tmp[n-1]| |vector_b[n-1]|
    # | ...                       bot[n]   mid[n] | | tmp[n] | | vector_b[n] |

    # allocate space for matrix
    top, mid = [0] * n, [0] * n
    vector_b = [0] * n
    n -= 1
    h = [x_val[i + 1] - x_val[i] for i in range(n)]

    # first row values
    top[0] = 0
    mid[0] = 1
    vector_b[0] = alpha

    # prepare vector b
    for i in range(1, n):
        part1 = 3 * (y_val[i + 1] - y_val[i]) / h[i]
        part2 = 3 * (y_val[i] - y_val[i - 1]) / h[i - 1]
        vector_b[i] = part1 - part2

    for i in range(1, n):
        mid[i] = 2 * (x_val[i + 1] - x_val[i - 1])  # 2 * (h[i] + h[i+1])
        mid[i] -= h[i - 1] * top[i - 1]  # sub row i-1 (to zero bot[i])

        # all elem on row_i are divided by mid[i] (after that mid[i] will be 1)
        top[i] = h[i] / mid[i]

        # sub in vector_b
        vector_b[i] -= h[i - 1] * vector_b[i - 1]
        vector_b[i] /= mid[i]

    # allocate space for coefficients
    a = y_val[:]
    b, d = [0] * n, [0] * n
    c = [0] * n + [beta]

    for i in range(n - 1, -1, -1):
        c[i] = vector_b[i] - top[i] * c[i + 1]
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
    fun = lambda x: x**2
    plot_function(fun, left, right, num=10)
