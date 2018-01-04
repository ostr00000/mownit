import numpy as np


def runge_kutta(fun, x_points, y0):
    size = len(x_points)
    y_values = np.zeros(size)
    y_values[0] = y0
    delta = (x_points[-1] - x_points[0]) / (size - 1)

    x = x_points[0]
    y = y0
    for i in range(size - 1):
        a = delta * fun(x, y)
        b = delta * fun(x + delta * 0.5, y + a * 0.5)
        c = delta * fun(x + delta * 0.5, y + b * 0.5)
        d = delta * fun(x + delta, y + c)

        y = y_values[i + 1] = y + (a + 2 * b + 2 * c + d) / 6
        x = x + delta

    return y_values
