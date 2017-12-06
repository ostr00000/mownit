from lab5.cubic_spline import derivative
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from itertools import zip_longest


def coef(x_val: List[float], y_val: List[float], z: float = 0.):
    n = len(x_val)
    assert n == len(y_val)
    n -= 1

    h = [x_val[i + 1] - x_val[i] for i in range(n)]
    vector_b = [z] + [2 * (y_val[i + 1] - y_val[i]) / h[i]
                      for i in range(n)]

    for i in range(1, n + 1):
        vector_b[i] -= vector_b[i - 1]

    b = vector_b[:-1]
    c = [(vector_b[i + 1] - vector_b[i]) / (2 * h[i]) for i in range(n)]

    a = []
    for i in range(n):
        y = (y_val[i] + y_val[i + 1]) / 2
        a.append(y - h[i] * (vector_b[i + 1] + vector_b[i]) / 4)

    return list(zip_longest(a, b, c, x_val[:]))


def spline_to_x_y(coef_and_x, num_of_points):
    n = len(coef_and_x) - 1
    spline_points = num_of_points / n
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

minim = 2
maxim = 10
x = range(minim,maxim)
y = range(minim+1,maxim+1)

x=[-1,0,1]
y =[0,1,3]

abc_x = coef(x, y, 0)
xx, yy = spline_to_x_y(abc_x, 100)
print(abc_x)
plt.plot(xx, yy, label="f(x)-interp")
# plt.plot()
plt.plot([minim, maxim], [0, 0], label="f(x)=0")
plt.scatter(x, y)
plt.legend()
plt.show()
