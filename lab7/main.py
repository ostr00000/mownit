from lab7.euler_method import *
from lab7.runge_kutta_method import *
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, NewType, List

Fun_der = NewType("Fun_der", Callable[[float, float], float])
Fun_method = NewType("Fun_method",
                     Callable[[Fun_der, List[float], float], List[float]])
Fun_original = NewType("Fun_original", Callable[[float], float])

k = 1
m = 1


def f(time, y_arg):
    total = k * m * y_arg * np.sin(m * time)
    total += k * k * m * np.sin(m * time) * np.cos(m * time)
    return total


def f_original(x_arg):
    return np.e ** (-k * np.cos(m * x_arg)) - k * np.cos(m * x_arg) + 1


def main_fun(type_method: Fun_method,
             y_der_function: Fun_der,
             y_der_original: Fun_original,
             y_start: float, left: float, right: float, div: int):
    x = np.linspace(left, right, div + 1)
    y_sol = type_method(y_der_function, x, y_start)
    y_ory = list(map(y_der_original, x))

    plt.plot([left, right], [0, 0], label="f(x)=0")
    plt.plot(x, y_sol, label="f(x) rozwiazanie numeryczne")
    plt.plot(x, y_ory, label="f(x) rozwiazanie dokładne")
    plt.legend()
    plt.show()


def newton_trial():
    def newton_cooling(_time, y_arg):
        return -0.07 * (y_arg - 20)

    def newton_cooling_original(x_arg):
        return 20 + (100 - 20) * np.e ** (-0.07 * x_arg)

    main_fun(euler, newton_cooling, newton_cooling_original,
             100, 0, 100, 10)


def runge_kutta_trial():
    def runge_kutta_fun(time, y_arg):
        return time * np.sqrt(y_arg)

    def runge_kutta_original(x_arg):
        return (x_arg ** 2 + 4) ** 2 / 16

    main_fun(runge_kutta, runge_kutta_fun, runge_kutta_original,
             1, 0, 10, 10)


if __name__ == "__main__":
    newton_trial()
