import matplotlib.pyplot as plt
from lab6.approximation import norm
import numpy as np
from typing import List


def approximate(degree: int, x_val: List[float],
                y_val: List[float]):
    n = len(x_val)
    assert n == len(y_val)
    assert n >= 2
    assert degree >= 0

    a0 = sum(y_val) / n
    a_coef, b_coef = [], []

    pin2 = np.pi * 2 / n

    # calc k coefficient
    for k in range(1, degree + 1):

        a_k, b_k = 0., 0.
        for i in range(n):
            trig_arg = k * i * pin2
            a_k += y_val[i] * np.cos(trig_arg)
            b_k += y_val[i] * np.sin(trig_arg)

        a_k *= 2
        a_k /= n
        b_k *= 2
        b_k /= n

        a_coef.append(a_k)
        b_coef.append(b_k)

    width = x_val[-1] - x_val[0]
    a_x = 2 * np.pi / width
    b_x = - np.pi * (x_val[0] + x_val[-1]) / width + np.pi

    def inner(x_arg):
        x_arg = a_x * x_arg + b_x
        total_val = a0
        for k, (a, b) in enumerate(zip(a_coef, b_coef), 1):
            total_val += a * np.cos(k * x_arg) + b * np.sin(k * x_arg)
        return total_val

    return inner


def fun(x):
    return x * np.sin(0.8 * np.pi / x)


if __name__ == "__main__":
    degree = 12
    left = 0.1
    right = 0.8
    points_num = 30

    degree = 11
    left = np.pi*-1
    right = np.pi*2
    points_num = 25

    def fun(x):
        return np.e**((-1)*2*np.sin(2*x))+2*np.sin(2*x)-1



    x = list(np.linspace(left, right, points_num))
    y = list(map(fun, x))

    approx = approximate(degree, x, y)

    xx = np.linspace(left, right, 1000)
    yy = list(map(fun, xx))
    yy_aprox = list(map(approx, xx))

    plt.plot(xx, yy, label="f(x)=...")
    plt.plot([left, right], [0, 0], label="f(x)=0")
    plt.scatter(x, y)
    plt.plot(xx, yy_aprox, label="f(x) aprox")

    plt.ylabel("y=f(x)")
    plt.xlabel("x watosci")
    plt.legend()
    plt.show()
