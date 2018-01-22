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


def norm(y_original, y_approximated):
    norm_max = 0
    norm_2 = 0
    for y1, y2 in zip(y_original, y_approximated):
        norm_max = max(norm_max, abs(y1 - y2))
        norm_2 += (y1 - y2) * (y1 - y2)
    return norm_max, norm_2


def main_fun(type_method: Fun_method,
             y_der_function: Fun_der,
             y_der_original: Fun_original,
             y_start: float, left: float, right: float, div: int):
    x = np.linspace(left, right, div + 1)
    x_real = np.linspace(left, right, (div+1)*100)
    y_sol = type_method(y_der_function, x, y_start)
    y_ory = list(map(y_der_original, x_real))
    points = list(map(y_der_original, x))

    nor = norm(y_sol, points)
    print("liczba kroków: {}, błąd maksimum: {}, średniokwadratowy: {}"
          .format(div, *nor))

    # print(y_sol)
    # print(y_ory)

    plt.plot([left, right], [0, 0], label="f(x)=0")
    plt.scatter(x, y_sol, label="f(x) rozwiazanie numeryczne punkty")
    plt.scatter(x, points,
                label="f(x) rozwiazanie dokładne punkty")
    plt.plot(x, y_sol, label="f(x) rozwiazanie numeryczne")
    plt.plot(x_real, y_ory, label="f(x) rozwiazanie dokładne")

    plt.title("Metoda Rungego-Kuty dla {} kroków".format(div))
    # plt.title("Metoda Eulera dla {} kroków".format(div))
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




def zad():
    x0 = - np.pi / 2
    xk = np.pi
    m = 4
    k = 1
    step = 20

    def fun_x_y(x, y):
        return (k ** 2 * m * np.sin(m * x) * np.cos(m * x)
                + k * m * y * np.sin(m * x))

    def fun_x(x):
        return np.e ** (-k * np.cos(m * x)) - k * np.cos(m * x) + 1

    a = fun_x(x0)

    main_fun(runge_kutta, fun_x_y, fun_x, a, x0, xk, step)

    # print("runge-kutta")
    # for step in range(20, 100):
    #     main_fun(runge_kutta, fun_x_y, fun_x, a, x0, xk, step)
    #
    #
    # print("euler")
    # for step in range(20, 100):
    #     main_fun(euler, fun_x_y, fun_x, a, x0, xk, step)



if __name__ == "__main__":
    zad()
