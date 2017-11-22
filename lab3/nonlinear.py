import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

DEF_EPS = 1e-2
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
    "x^10-(1-x)^15": lambda x: x ** 10 - (1 - x) ** 15,
    "x^2": lambda x: x * x,
    "x^2-4": lambda x: x * x - 4,
    "x^6+x": lambda x: x ** 6 + x,
    "(x-1)e^-2x+x^2": lambda x: (x - 1) * np.e ** (-2 * x) + x * x,
}

end_functions = {
    "|x(i+1)-x(i)|<p": end_function_1,
    "|f(x(i))|<p": end_function_2,
}

methods = {
    "metoda siecznych": secant_method,
    "metoda newtona": newton_method,
}


def end_functions_inv(end):
    return next(k for k, v in end_functions.items() if v == end)


def methods_inv(met):
    return next(k for k, v in methods.items() if v == met)


def _find(epsilon, method_str, end_function_str, inc=True):
    inc_str = "zwiekszanie" if inc else "zmniejszanie"
    logger.info(method_str + end_function_str + " " + inc_str)

    fun_name = "x^10-(1-x)^15"
    f = functions[fun_name]
    start_x = 0.1
    end_x = 2.1
    results = []
    pattern = "miejce zerowe: {} w iteracjach {}" + \
              (" parametry startowe: {:2.1f} oraz {:2.1f} "
               if method_str == "metoda siecznych"
               else " parametr startowy {:2.1f}")
    method = methods[method_str]
    end_function = end_functions[end_function_str]

    for _ in range(20):
        zero, iteration = method(f, end_function(f, epsilon), start_x, end_x)
        logger.info(pattern.format(zero, iteration, start_x, end_x))
        results.append((start_x if inc else end_x, zero, iteration))

        if inc:
            start_x += 0.1
        else:
            end_x -= 0.1

    return results


def find(eps, end):
    end1 = "|x(i+1)-x(i)|<p"
    end2 = "|f(x(i))|<p"
    dec = " zmniejszajac x"
    sec = "metoda siecznych"
    newt = "metoda newtona"

    data_dict = {
        sec + " " + end1: _find(eps, sec, end1),
        sec + " " + end2: _find(eps, sec, end2),
        sec + " " + end1 + dec: _find(eps, sec, end1, inc=False),
        sec + " " + end2 + dec: _find(eps, sec, end2, inc=False),
        newt + " " + end1: _find(eps, newt, end1),
        newt + " " + end2: _find(eps, newt, end2),
    }

    def plot(name, data, t='i'):
        start_point, zero, iteration = list(zip(*data))
        plt.clf()
        plt.plot(start_point, iteration if t == 'i' else zero, label=name)
        plt.legend(bbox_to_anchor=(0., 1.05, 1., .102), loc=3,
                   mode="expand", borderaxespad=0.)
        # locs, labels = plt.yticks()
        # locs = [0.42, 0.43, 0.44]
        # plt.yticks(locs, map(lambda x: "{:.2f}".format(x), locs))
        plt.ylabel("liczba iteracji" if t == 'i' else "obliczony punkt zerowy")
        plt.xlabel("punkt startowy")
        plt.tight_layout()

        if end:
            plt.savefig("data/" + name + ".pdf")
            # plt.show()

    for name, data in data_dict.items():
        plot(name, data, 'h')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    function_name = "x^10-(1-x)^15"
    fun = functions[function_name]

    # for epsilon in (1e-14,):
    #     find(epsilon, True)

    plt.subplot(111)
    x = np.linspace(0.1, 2.1, 100)
    y = fun(x)

    plt.xlabel("zmienna x")
    plt.ylabel("wartość zmiennej y")
    plt.plot(x, y, label="f(x)="+function_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig("data/" + "zad1" + ".png")
    # plt.show()