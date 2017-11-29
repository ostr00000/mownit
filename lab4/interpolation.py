import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


def newton_interpolation(x_points: List[float], y_points: List[float]):
    a = [i for i in y_points]
    for j in range(1, len(x_points)):
        for i in range(len(x_points) - 1, j - 1, -1):
            a[i] = (a[i] - a[i - 1]) / float(x_points[i] - x_points[i - j])

    def inner(x_arg):
        f_x_arg = a[-1]
        for i in range(len(x_points) - 2, -1, -1):
            f_x_arg = f_x_arg * (x_arg - x_points[i]) + a[i]
        return f_x_arg

    return inner


def lagrange_interpolation(x_points: List[float], y_points: List[float]):
    def inner(x_arg):
        f_x_arg = 0
        for x, y in zip(x_points, y_points):
            mul = 1
            for xx, yy in zip(x_points, y_points):
                if xx == x:
                    continue
                mul *= (x_arg - xx) / float(x - xx)
            f_x_arg += y * mul
        return f_x_arg

    return inner


def norm(y_original, y_interpolated):
    norm_max = 0
    norm_2 = 0
    for y1, y2 in zip(y_original, y_interpolated):
        norm_max = max(norm_max, abs(y1 - y2))
        norm_2 += (y1 - y2) * (y1 - y2)
    return norm_max, norm_2


def zeros_chebyshev(k):
    return [np.cos((2 * j - 1) / (2 * k) * np.pi) for j in range(k, 0, -1)]


def plot_functions(fun_name, method=lagrange_interpolation,
                   left=-1., right=1., nodes=10, use_cheb_roots=False,
                   flg_last=False):
    fun = functions[fun_name]()

    # original function
    x = np.linspace(left, right, 1000)
    y = list(map(fun, x))

    # prepare to interpolate
    if use_cheb_roots:
        def scale(x):
            return x * (right - left) / 2 + (left + right) / 2

        x_points = list(map(scale, zeros_chebyshev(nodes)))
    else:
        x_points = list(np.linspace(left, right, nodes))
    y_points = list(map(fun, x_points))
    lag = method(x_points, y_points)
    y_interpolated = list(map(lag, x))

    int_name = "lagrangea" if method == lagrange_interpolation else "newtona"
    nodes_type = "Czebyszewa" if use_cheb_roots else "rownoodległych"
    logger.info("metoda interpolacji {} dla {} węzłów {}"
                .format(int_name, nodes, nodes_type))
    logger.info("dokładność: średniokwadratowa (z podziałem na 1001 punktów): {}, w metryce maximum: {} "
                .format(*norm(y_points, y_interpolated)))

    # plot
    plt.subplot(111)
    plt.plot(x, y_interpolated,
             label=fun_name + " dla " + str(nodes) + " interpolowanych węzłów")
    plt.scatter(x_points, y_points)

    if flg_last:
        plt.plot(x, y, label=fun_name)
        plt.plot(x, x * 0, label="f(x)=0")
        plt.legend()
        plt.show()


def simple(k=None, m=None):
    return lambda x: sum(x ** n for n in range(5))


def f1(k=0.8, m=None):
    def inner(x):
        return x * np.sin(k * np.pi / x)

    return inner


functions = {
    "x^5": lambda k, m: lambda x: x ** 5,
    "test function": simple,
    "x*sin(0.8*pi/x)": f1,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    fun_name = "x*sin(0.8*pi/x)"
    max_nodes = 20
    plot_functions(fun_name, left=0.1, right=0.8, nodes=5, flg_last=True, use_cheb_roots=True)
    # for i in range(3, max_nodes):
    #     plot_functions(fun_name, left=0.1, right=0.8, nodes=i,
    #                    flg_last=max_nodes - 1 == i, use_cheb_roots=True)
        # plot_functions(fun_name, left=0.1, right=0.8, nodes=i,
        #            flg_last=True, use_cheb_roots=True)
