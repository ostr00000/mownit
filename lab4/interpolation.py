import matplotlib.pyplot as plt
import numpy as np
import logging
from typing import List, Callable

logger = logging.getLogger(__name__)
EPS = 1e-4


def derivative(func, eps=EPS):
    def inner(x, n=1):
        if n == 1:
            return (func(x + eps) - func(x - eps)) / eps / 2
        return (inner(x + eps, n - 1) - inner(x - eps, n - 1)) / eps / 2

    return inner


def hermite_interpolation(x_points: List[float], y_points: List[float],
                          fun_der: Callable[[float, int], float]):
    a = [i for i in y_points]
    factorial = 1
    for j in range(1, len(x_points)):
        factorial *= j
        for i in range(len(x_points) - 1, j - 1, -1):
            if x_points[i] == x_points[i - j]:
                a[i] = fun_der(x_points[i], j) / factorial
            else:
                a[i] = (a[i] - a[i - 1]) / float(x_points[i] - x_points[i - j])

    def inner(x_arg):
        f_x_arg = a[-1]
        for i in range(len(x_points) - 2, -1, -1):
            f_x_arg = f_x_arg * (x_arg - x_points[i]) + a[i]
        return f_x_arg

    return inner


def newton_interpolation(x_points: List[float], y_points: List[float], _=None):
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


def lagrange_interpolation(x_points: List[float], y_points: List[float],
                           _=None):
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
                   flg_last=False, type=None):
    fun = functions[fun_name]()

    # original function
    x = np.linspace(left, right, 1000)
    y = list(map(fun, x))

    # prepare to interpolate points
    if not type:
        if use_cheb_roots:
            def scale(x):
                return x * (right - left) / 2 + (left + right) / 2

            x_points = list(map(scale, zeros_chebyshev(nodes)))
        else:
            x_points = list(np.linspace(left, right, nodes))

    elif (type == "test" and fun_name == "x^8+1"
          and method == hermite_interpolation):
        x_points = [-1] * 3 + [0] * 3 + [1] * 3

    else:
        def gen():
            if nodes % 2:
                yield left
            for x in np.linspace(left, right, nodes // 2):
                yield x
                yield x

        x_points = list(gen())

    # interpolate nodes
    y_points = list(map(fun, x_points))
    lag = method(x_points, y_points, derivative(fun))
    y_interpolated = list(map(lag, x))

    # print text information
    texts = {
        lagrange_interpolation: ("lagrange'a",),
        newton_interpolation: ("newtona",),
        hermite_interpolation: ("hermita",),
    }
    nodes_type = " zer wielomianu Czebyszewa" if use_cheb_roots \
        else "rownoodległych"
    logger.info("metoda interpolacji {} dla {} węzłów {}"
                .format(texts[method][0], nodes, nodes_type))
    logger.info("dokładność: średniokwadratowa (z podziałem na 1001 punktów):"
                " {}, w metryce maximum: {} "
                .format(*norm(y, y_interpolated)))

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


def simple(_k=None, _m=None):
    return lambda x: sum(x ** n for n in range(5))


def f1(k=0.8, _m=None):
    def inner(x):
        return x * np.sin(k * np.pi / x)

    return inner


functions = {
    "x^5": lambda _k, _m: lambda x: x ** 5,
    "test function": simple,
    "x*sin(0.8*pi/x)": f1,
    "x^8+1": lambda _k=None, _m=None: lambda x: x ** 8 + 1,
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    fun_name = "x*sin(0.8*pi/x)"
    # max_nodes = 20
    # plot_functions(fun_name, left=0.1, right=0.8, nodes=5, flg_last=True,
    #                use_cheb_roots=True)

    # # test from wikipedia
    # fun_name = "x^8+1"
    # hermit_plot(fun_name, left=-1, right=1, nodes=9, type="test")

    # hermit_plot(fun_name, left=0.1, right=0.8, nodes=30, use_cheb_roots=True)

    max_nodes = 48
    for i in range(47, max_nodes):
        plot_functions(fun_name, method=hermite_interpolation,
                       left=0.1, right=0.8, nodes=i,
                       flg_last=max_nodes - 1 == i, use_cheb_roots=False)
        # plot_functions(fun_name, left=0.1, right=0.8, nodes=i,
        #                flg_last=True, use_cheb_roots=True)

    #hermita: 47 równoodległe, 34 czebyszewa