import lab1.precision as precision
import lab1.matrix_gauss as mg
import logging
import copy
import matplotlib.pyplot as plt
from time import time
from numpy import dtype


logger = logging.getLogger('matrix')
logger.setLevel(logging.INFO)

number = precision.number


def a3(i, j, m=2, k=5):
    if j == i:
        return number(-number(m * i) - number(k))
    if j == i + 1:
        return number(i)
    if j == i - 1:
        return number(number(m) / number(i))
    else:
        return number(0)


def get_tri_diagonal_matrix(fun, n):
    assert n >= 2
    top, mid, bot = [None]*n, [None]*n, [None]*n

    for i in range(1, n+1):
        top[i-1] = fun(i, i+1) if i != n else None
        mid[i-1] = fun(i, i)
        bot[i-1] = fun(i, i-1) if i != 1 else None

    return top, mid, bot


def thomas_algorithm(tri_diagonal_matrix, vec):
    top, mid, bot = copy.deepcopy(tri_diagonal_matrix)
    matrix_size = len(vec)
    assert matrix_size == len(top)
    assert matrix_size == len(mid)
    assert matrix_size == len(bot)

    # decomposition
    for i in range(1, matrix_size):
        bot[i] = number(bot[i] / mid[i-1])
        mid[i] = number(mid[i] - number(bot[i] * top[i-1]))

    # forward substitution
    new_vec = [vec[0][0]]
    for i in range(1, matrix_size):
        new_vec.append(number(vec[i][0] - number(bot[i] * new_vec[i-1])))

    # back substitution
    x = [None] * matrix_size
    x[matrix_size-1] = [number(new_vec[matrix_size-1] / mid[matrix_size-1])]
    for i in range(matrix_size-2, -1, -1):
        x[i] = [number(number(new_vec[i] - number(top[i] * x[i+1][0])) / mid[i])]

    return x


def get_size(obj):
    if isinstance(obj, tuple):
        # three list in tuple with size of float number
        return 3 * len(obj[0]) * dtype(number).itemsize

    elif isinstance(obj, list):
        # n * n elements with size of float number
        return len(obj) * len(obj) * dtype(number).itemsize

    else:
        raise TypeError


def measurement(fun, a, b, x, name):
    t0 = time()
    result = fun(a, b)
    t1 = time()

    distance = mg.get_vector_distance(x, result)
    total_time = t1 - t0
    structure_size = get_size(a)

    logger.debug("{} result : {}".format(name, result))
    logger.info("{} maximum metric : {}".format(name, distance))
    logger.info("{} time : {}".format(name, total_time))
    logger.info("{} structure size : {}".format(name, structure_size))

    return distance, total_time, structure_size


def gauss_and_thomas(fun, n):
    A = mg.get_matrix(fun, n)
    AA = get_tri_diagonal_matrix(fun, n)
    x = mg.get_vertical_vector_x(n)
    B = mg.matrix_mul(A, x)

    logger.debug("A:  {}".format(A))
    logger.debug("AA: {}".format(AA))
    logger.debug("x:  {}".format(x))
    logger.debug("B:  {}".format(B))

    gauss = measurement(mg.gauss_elimination, A, B, x, "Gauss")
    thomas = measurement(thomas_algorithm, AA, B, x, "Thomas")

    return gauss, thomas


def experiment(fun, max_size=50):
    data = {}

    for float_name, float_type in precision.floating_point_types.items():
        global number
        number = float_type
        mg.number = float_type
        data[float_name] = []

        logger.info("precision type: {}".format(float_name))

        for n in range(2, max_size + 1):
            logger.info("matrix size {}".format(n))
            data[float_name].append(gauss_and_thomas(fun, n))

        logger.info("")

    def plot_data(tuple_num, filename, is_log=False, label="", title=""):
        def gen(float_num, is_gauss):
            alg_type = 0 if is_gauss else 1
            gen_data = data[float_num]
            for i in range(len(gen_data)):
                yield gen_data[i][alg_type][tuple_num]

        plt.subplot(111)
        plt.title(title)
        plt.ylabel(label)
        plt.xlabel("Rozmiar macierzy A")
        if is_log:
            plt.semilogy()

        for float_n in precision.floating_point_types.keys():
            plt.plot(range(2, max_size + 1),
                     list(gen(float_n, True)),
                     label=float_n + ' gauss')
            plt.plot(range(2, max_size + 1),
                     list(gen(float_n, False)),
                     label=float_n + ' thomas')

        #plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    plot_data(0, "distance.pdf", True, "Błąd w metryce maksimum",
              "Funkcja błędu algorythmu Thomasa od wielkości macierzy")
    plot_data(1, "time.pdf", False, "Czas obliczania [s]",
              "Funkcja długości czasu obliczania od wielkości macierzy")
    plot_data(2, "size.pdf", False, "Rozmiar struktury [bajt]",
              "Rozmiar struktury danych od wielkości macierzy")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    experiment(a3, 100)
