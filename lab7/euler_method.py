import numpy as np


def euler(fun, x_points, y0):
    size = len(x_points)
    y_values = np.zeros(size)
    y_values[0] = y0
    delta = (x_points[-1] - x_points[0]) / (size - 1)

    for i in range(size - 1):
        y_values[i + 1] = delta * fun(x_points[i], y_values[i]) + y_values[i]

    return y_values
