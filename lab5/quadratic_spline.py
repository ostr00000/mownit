import matplotlib.pyplot as plt
import numpy as np
from typing import List, NamedTuple
from itertools import zip_longest
from lab5.cubic_spline import derivative


class BoundCondition(NamedTuple):
    is_left: bool = True
    value: float = 0.


def coef(x_val: List[float], y_val: List[float],
         bound_condition: BoundCondition = BoundCondition(True, 0.)):
    n = len(x_val)
    assert n == len(y_val)
    n -= 1

    h = [x_val[i + 1] - x_val[i] for i in range(n)]
    vector_b = [2 * (y_val[i + 1] - y_val[i]) / h[i]
                for i in range(n)]

    if bound_condition.is_left:
        vector_b = [bound_condition.value] + vector_b
        for i in range(1, n + 1):
            vector_b[i] -= vector_b[i - 1]
    else:
        vector_b = vector_b + [bound_condition.value]
        for i in range(n, 0, -1):
            vector_b[i - 1] -= vector_b[i]

    b = vector_b[:-1]
    c = [(vector_b[i + 1] - vector_b[i]) / (2 * h[i]) for i in range(n)]

    a = []
    for i in range(n):
        y = (y_val[i] + y_val[i + 1]) / 2
        a.append(y - h[i] * (vector_b[i + 1] + vector_b[i]) / 4)

    return list(zip_longest(a, b, c, x_val[:]))


def spline_to_x_y(coef_and_x, num_of_points):
    n = len(coef_and_x) - 1
    spline_points = num_of_points // n
    if spline_points < 2:
        spline_points = 2

    x, y = [], []
    for i in range(n):
        a, b, c, x0 = coef_and_x[i]
        x1 = coef_and_x[i + 1][3]

        for xi in np.linspace(x0, x1, spline_points):
            x.append(xi)
            h = xi - x0
            y.append(a + b * h + c * h ** 2)

    return x, y


if __name__ == "__main__":
    left = 0.1
    right = 0.8
    points_num = 10

    def fun(x):
        return x * np.sin(0.8 * np.pi / x)

    x = list(np.linspace(left, right, points_num))
    y = list(map(fun, x))

    abc_x = coef(x, y, BoundCondition(value=derivative(fun)(left)))
    xx, yy = spline_to_x_y(abc_x, 100 * points_num)

    org_x = list(np.linspace(left, right, points_num ** 2 * 10))
    org_y = list(map(fun, org_x))

    plt.plot(xx, yy, label="f(x)-interp")
    plt.plot(org_x, org_y, label="f(x)-org")
    plt.plot([left, right], [0, 0], label="f(x)=0")
    plt.scatter(x, y)
    plt.legend()
    plt.show()
