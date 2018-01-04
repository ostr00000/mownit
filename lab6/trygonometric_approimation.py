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

    pin2 = np.pi * 2 / (n-1)

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


    # f(x) = (new_r-new_l)/(old_r-old_l)*(x-(old_l+old_r)/2)+(new_l+new_r)/2

    width = x_val[-1] - x_val[0]
    a_x = 2 * np.pi / width
    b_x = (x_val[0] + x_val[-1]) / 2

    def inner(x_arg):
        x_arg = a_x * (x_arg - b_x) + np.pi
        total_val = a0
        for k, (a, b) in enumerate(zip(a_coef, b_coef), 1):
            total_val += a * np.cos(k * x_arg) + b * np.sin(k * x_arg)
        return total_val

    return inner


def fun(x):
    return x * np.sin(0.8 * np.pi / x)


def iterate_fun():
    for i in range(1, 40):
        degree = i
        x = list(np.linspace(left, right, points_num))
        y = list(map(fun, x))

        approx = approximate(degree, x, y)

        xx = np.linspace(left, right, 1000)
        yy = list(map(fun, xx))
        yy_aprox = list(map(approx, xx))

        print(norm(yy, yy_aprox))


if __name__ == "__main__":
    degree = 5
    left = 0.1
    right = 0.8
    # points_num = 40

    for i in range(degree, 50):
        points_num = i

        x = list(np.linspace(left, right, points_num))
        y = list(map(fun, x))

        approx = approximate(degree, x, y)

        xx = np.linspace(left, right, 1000)
        yy = list(map(fun, xx))
        yy_aprox = list(map(approx, xx))

        print("{} {:2.4f} {:2.4f}".format(i, *norm(yy, yy_aprox)))

        if i == 20:
            plt.title("Liczba węzłów: 20, stopien wielomianu: 5")
            plt.plot(xx, yy, label="f(x)=...")
            plt.plot([left, right], [0, 0], label="f(x)=0")
            plt.scatter(x, y)
            plt.plot(xx, yy_aprox, label="f(x) aprox")

            plt.ylabel("y=f(x)")
            plt.xlabel("x watosci")
            plt.legend()
            plt.show()
