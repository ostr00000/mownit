import matplotlib.pyplot as plt
import numpy as np
from typing import List, NamedTuple
from itertools import zip_longest
from lab5.cubic_spline import derivative


def norm(y_original, y_interpolated):
    norm_max = 0
    norm_2 = 0
    for y1, y2 in zip(y_original, y_interpolated):
        norm_max = max(norm_max, abs(y1 - y2))
        norm_2 += (y1 - y2) * (y1 - y2)
    return norm_max, norm_2


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
    points_num = 5

    def fun(x):
        return x * np.sin(0.8 * np.pi / x)


    data = []
    for i in range(3, 200):
        points_num = i
        x = list(np.linspace(left, right, points_num))
        y = list(map(fun, x))

        abc_x = coef(x, y, BoundCondition(value=derivative(fun)(right), is_left=False))
        xx, yy = spline_to_x_y(abc_x, 1000)

        org_x = list(np.linspace(left, right, 1000))
        org_y = list(map(fun, xx))

        data.append((i, *norm(org_y, yy)))
        # print(data[-1])

        if 15 == i:
            fig = plt.figure(figsize=(10, 20), dpi=100)
            ax = fig.add_subplot(111)

            plt.plot(xx, yy, label="f(x) funkcja interpolujaca")

            ax.plot(xx, org_y, label="f(x) = x * sin(0.8 * pi / x)")
            ax.plot([left, right], [0, 0], label="f(x)=0")
            ax.legend()

            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.scatter(x, y)
            plt.show()
            fig.savefig("test.pdf")


    # data = list(zip(*data))
    #
    # fig = plt.figure(figsize=(10, 20), dpi=100)
    # ax = fig.add_subplot(111)
    # ax.plot(data[0], data[2], label="błąd średniokwadratoey")
    # ax.plot(data[0], data[1], label="błąd w metryce maksimum")
    # ax.set_xlabel("liczba węzłów", size=15)
    # ax.semilogy()
    # ax.set_ylabel("wartość błędu", size=15)
    # ax.legend()
    # plt.show()
    # fig.savefig("test.pdf")


