import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

DEF_EPS = 1e-10
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


functions = {
    "x^10-(1-x)^15": lambda x: x**10-(1-x)**15,
    "x^2": lambda x: x*x,
    "x^2-4": lambda x: x*x-4,
    "x^6+x": lambda x: x**6+x,
    "(x-1)e^-2x+x^2": lambda x: (x-1)*np.e**(-2*x)+x*x,
}

def find(eps, end):
    data = {}
    fun_name = "x^10-(1-x)^15"
    f = functions[fun_name]
    pattern = "miejce zerowe: {} w iteracjach {}, parametry startowe: {} oraz 2.1 "
    logger.info("\n\n")
    logger.info("obliczenia dla funkcji: " + fun_name
                + " w przedziale [0.1, 2.1] dla epsilon={}".format(eps))

    tmp = []
    logger.info("metoda siecznych |x(i+1)-x(i)|<p - zwiekszanie")
    x = 0.1
    for _ in range(20):
        zero, iteration = secant_method(f, end_function_1(f,  eps), x, 2.1)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x += 0.1
    data['s1'] = tmp

    tmp = []
    logger.info("metoda siecznych |f(x(i))|<p - zwiekszanie")
    x = 0.1
    for _ in range(20):
        zero, iteration = secant_method(f, end_function_2(f, eps), x, 2.1)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x += 0.1
    data['s2'] = tmp


    pattern = "miejce zerowe: {} w iteracjach {}, parametry startowe: 0.1 oraz {}"

    logger.info("metoda siecznych |x(i+1)-x(i)|<p - zmniejszanie")
    x = 2.1
    tmp = []
    for _ in range(20):
        zero, iteration = secant_method(f, end_function_1(f, eps), 0.1, x)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x -= 0.1
    data["s1d"] = tmp

    logger.info("metoda siecznych |f(x(i))|<p - zmniejszanie")
    x = 2.1
    tmp = []
    for _ in range(20):
        zero, iteration = secant_method(f, end_function_2(f, eps), 0.1, x)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x -= 0.1
    data['s2d'] = tmp

    pattern = "miejce zerowe: {} w iteracjach {}, parametr startowy: {}"

    logger.info("metoda newtona |x(i+1)-x(i)|<p")
    x = 0.1
    tmp = []
    for _ in range(20):
        zero, iteration = newton_method(f, end_function_1(f, eps), x)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x += 0.1
    data["n1"] = tmp

    logger.info("metoda newtona |f(x(i))|<p ")
    x = 0.1
    tmp = []
    for _ in range(20):
        zero, iteration = newton_method(f, end_function_2(f, eps), x)
        logger.info(pattern.format(zero, iteration, x))
        tmp.append((x, iteration))
        x += 0.1
    data['n2'] = tmp



    def get_id(d, id):
        for dd in d:
            yield dd[id]

    def pl(d, name):
        plt.plot(list(get_id(d, 0)), list(get_id(d, 1)), label=name)
        plt.legend(loc='upper left')
        plt.ylabel("liczba iteracji")
        plt.xlabel("punkt startowy")
        if end:
            plt.savefig(name + ".pdf")
        #plt.show()


    pl(data["n1"], "newton x1-x2 {}".format(eps))





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    function_name = "x^10-(1-x)^15" # "(x-1)e^-2x+x^2"
    fun = functions[function_name]

    e = 1e-4
    for w in range(4):
        find(e, w == 3)
        e /= 100

    # x_0, n_it = secant_method(fun, end_function_1(fun), -12., 11.)
    # logger.info("x_0: {}, iterations: {}".format(x_0, n_it))
    #
    # x_0, n_it = newton_method(fun, end_function_1(fun), -12., 11.)
    # logger.info("x_0: {}, iterations: {}".format(x_0, n_it))
    #
    # xx = np.linspace(0.1, 2.1, 20)
    # yy = list(fun(i) for i in xx)

    # plt.plot(xx, yy, label="f(x)=" + function_name)
    # plt.plot((0.1, 2.1), (0, 0), label="f(x)=0")
    # plt.legend(loc='upper left')
    # plt.show()
