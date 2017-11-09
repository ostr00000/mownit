import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

DEF_EPS = 1e-6
MAX_ITER = 100


def end_function_1(_, epsilon=DEF_EPS):
    def end(old_x, new_x):
        return abs(old_x - new_x) < epsilon
    return end


def end_function_2(fun, epsilon=DEF_EPS):
    def end(_, new_x):
        return abs(fun(new_x)) < epsilon
    return end


def derivative(fun, epsilon=DEF_EPS):
    def der(x):
        return (fun(x + epsilon) - fun(x)) / epsilon
    return der


def newton_method(fun, end_function, start_x, _=None):
    der = derivative(fun)

    iterations = 0
    old_x = start_x
    while True:
        iterations += 1
        new_x = old_x - fun(old_x) / der(old_x)

        if end_function(old_x, new_x) or iterations > MAX_ITER:
            return new_x, iterations

        old_x = new_x


def secant_method(fun, end_function, start_a, start_b):
    iterations = 0

    f_b = fun(start_b)
    while True:
        iterations += 1

        f_a = fun(start_a)
        new_x = start_a - (start_a - start_b) / (f_a - f_b) * f_a

        if end_function(new_x, start_a) or iterations > MAX_ITER:
            return new_x, iterations

        start_b = start_a
        start_a = new_x
        f_b = f_a



def a1(x):
    return x*x-4


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    fun = a1
    x_0, n_it = secant_method(fun, end_function_1(fun), -12., 11.)
    logger.info("x_0: {}, iterations: {}".format(x_0, n_it))

    x = np.linspace(-10, 10, 1000)
    y = list(fun(i) for i in x)

    plt.plot(x, y, label="f(x)=x^2+4")
    plt.plot((-10, 10), (0, 0), label="f(x)=0")
    plt.legend(loc='lower left')
    plt.show()
