import logging
import lab2.matrix_util as mu
import numpy as np
import matplotlib.pyplot as plt
from time import time
from typing import Callable, Tuple, List

Mat = List[List]
Method_stats = Tuple[Mat, int, float]
End_fun_ret = Callable[[Mat, Mat], bool]
End_fun_sig = Callable[[float, Mat, Mat], End_fun_ret]
Method_signature = Callable[[Mat, Mat, End_fun_ret, Mat, float], Method_stats]

logger = logging.getLogger(__name__)
MAX_ITER = 200
MAX_POW_EPS = 24
MAX_INIT_NUM_POW = 15
DEFAULT_EPSILON = 1e-6
DEFAULT_MATRIX_SIZE = 100
SOR_SCALE = 10


def get_max_efficient_factor(radius):
    return 2 / (1 + np.sqrt(1 - radius * radius))


def end_function_1(epsilon: float, _: Mat, __: Mat):
    def end(old_vec: Mat, new_vec: Mat) -> bool:
        return mu.get_distance(old_vec, new_vec) < epsilon

    return end


def end_function_2(epsilon: float, matrix_A: Mat, vector_b: Mat):
    def end(_, new_vec: Mat) -> bool:
        matrix_Ax = mu.matrix_mul(matrix_A, new_vec)
        distance = mu.get_distance(matrix_Ax, vector_b)
        return distance < epsilon

    return end


def prepare_const_values(matrix: Mat, vector: Mat,
                         factor: float = 1.) -> Tuple[Mat, Mat]:
    matrix_size = len(matrix)
    inverted = [(float(factor) / matrix[i][i]) for i in range(matrix_size)]
    matrix = [[(-matrix[i][j] * inverted[i]) if i != j else 0
               for j in range(matrix_size)]
              for i in range(matrix_size)]

    inverted = [vector[i][0] * inverted[i] for i in range(matrix_size)]
    return matrix, inverted


def iterate_algorithm(matrix: Mat, vector: Mat,
                      end_condition: Callable[[Mat, Mat], bool],
                      step_function: Callable[[Mat, Mat, Mat], Mat],
                      initial_guess: Mat = None,
                      factor: float = 1.) -> Tuple[Mat, int, float]:
    matrix_size = len(matrix)
    assert matrix_size == len(matrix[0])
    assert matrix_size == len(vector)

    # prepare const values
    matrix, inverted = prepare_const_values(matrix, vector, factor)
    old_vector = initial_guess or [[0] for _ in range(matrix_size)]

    # spectral radius
    spectral_rad = max(abs(np.linalg.eigvals(np.array(matrix))))
    logger.info("spectral radius: {}".format(spectral_rad))
    if spectral_rad >= 1:
        logger.warning("spectral radius greater than 1")

    # start iterate
    iteration_num = 0
    while True:
        iteration_num += 1
        new_vector = step_function(matrix, inverted, old_vector)

        # check end condition
        if end_condition(old_vector, new_vector) or iteration_num > MAX_ITER:
            return new_vector, iteration_num, spectral_rad
        else:
            old_vector = new_vector


def jacobi_method(matrix, vector, end_condition,
                  initial_guess=None, _=None) -> Method_stats:
    def step_function(mat: Mat, inverted: Mat, old_vector: Mat) -> Mat:
        matrix_size = len(mat)
        new_vector = []
        for i in range(matrix_size):
            total = sum((old_vector[j][0] * mat[i][j])
                        for j in range(matrix_size))
            new_vector.append([inverted[i] + total])
        return new_vector

    return iterate_algorithm(matrix, vector, end_condition, step_function,
                             initial_guess=initial_guess)


def gauss_seidel(matrix, vector, end_condition,
                 initial_guess=None, _=None) -> Method_stats:
    def step_function(mat: Mat, inverted: Mat, old_vector: Mat) -> Mat:
        matrix_size = len(mat)
        new_vector = []
        for i in range(matrix_size):
            total_before = sum((old_vector[j][0] * mat[i][j])
                               for j in range(i + 1, matrix_size))

            total_after = sum((new_vector[j][0] * mat[i][j])
                              for j in range(0, i))
            new_vector.append([inverted[i] + total_before + total_after])
        return new_vector

    return iterate_algorithm(matrix, vector, end_condition, step_function,
                             initial_guess=initial_guess)


def successive_over_relaxation(matrix, vector, end_condition,
                               initial_guess=None,
                               factor=1.1139429023573001) -> Method_stats:
    def step_function(mat: Mat, inverted: Mat, old_vector: Mat) -> Mat:
        matrix_size = len(mat)
        new_vector = []
        for i in range(matrix_size):
            total_before = sum((old_vector[j][0] * mat[i][j])
                               for j in range(i + 1, matrix_size))

            total_after = sum((new_vector[j][0] * mat[i][j])
                              for j in range(0, i))

            val_next = inverted[i] + total_before + total_after
            val_prev = (1 - factor) * old_vector[i][0]
            new_vector.append([val_next + val_prev])
        return new_vector

    return iterate_algorithm(matrix, vector, end_condition, step_function,
                             initial_guess=initial_guess, factor=factor)


def a1(n: int, k: int = 11, m: int = 2) -> Callable[[int, int], float]:
    def inner(i: int, j: int) -> float:
        if i == j:
            return k
        else:
            return m / (n - i - j + 0.5)

    return inner


def single_test(matrix_size: int,
                end_fun: End_fun_sig,
                epsilon: float = DEFAULT_EPSILON,
                factor: float = 1.,
                initial_guess: Mat = None,
                method: Method_signature = jacobi_method):
    A = mu.get_matrix(a1(matrix_size), matrix_size)
    x = mu.get_vertical_vector_x(matrix_size)
    B = mu.matrix_mul(A, x)

    logger.debug("A: {}".format(A))
    logger.debug("x: {}".format(x))
    logger.debug("B: {}".format(B))

    end = end_fun(epsilon, A, B)

    time0 = time()
    X, iterations, rad = method(A, B, end, initial_guess, factor)
    time1 = time()

    distance = mu.get_distance(X, x)
    total_time = time1 - time0

    logger.debug("X: {}".format(X))
    logger.debug("Błąd w metryce maksimu{}".format(distance))
    logger.info("result: {} in {} iterations\n".format(distance, iterations))
    return distance, iterations, rad, total_time


def plot(data: List[Method_stats], tuple_id: int,
         name: str, x_label: str, x_log: bool = False, range_gen=None):
    range_gen = range_gen or range(1, len(data) + 1)

    name = "Funkcja " + (" odległości",
                         " liczby iteracji",
                         " promienia spektralnego")[tuple_id] + name
    plt.clf()
    plt.figure(num=1, figsize=(10, 20), dpi=100)
    # plt.title(name)
    plt.ylabel(("Odległość w metyce maksimum",
                "Liczba iteracji",
                "Promień spektralny")[tuple_id])
    plt.xlabel(x_label)
    if tuple_id == 0:
        plt.semilogy()
    if x_log:
        plt.semilogx()

    plt.plot(range_gen, list(d[tuple_id] for d in data))

    # plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig("data/" + name + ".pdf")
    # plt.show()


end_functions = {
    end_function_1: "|x(i+1)-x(i)|<eps",
    end_function_2: "|Ax(i)-b|<eps"
}

methods = {
    'Jacobi': jacobi_method,
    'SOR': successive_over_relaxation,
    'Gauss-Seidel': gauss_seidel,
}


def methods_inv(method):
    return next(k for k, v in methods.items() if v == method)


def test_size(end_fun: End_fun_sig, method: Method_signature):
    logger.info("size test")
    data = []
    for i in range(1, 100):
        logger.info("size: {}".format(i))
        data.append(single_test(i, end_fun, method=method))

    name = " od rozmiaru macierzy z warunkiem stopu {} dla metody {}" \
        .format(methods_inv(method), end_functions[end_fun])
    x_label = "Rozmiar macierzy A"

    for i in range(3):
        plot(data, i, name, x_label)


def test_vector(end_fun: End_fun_sig, method: Method_signature):
    logger.info("vector test")
    data = []
    range_gen = list(2 ** j for j in range(0, MAX_INIT_NUM_POW))
    for i in range_gen:
        logger.info("vector: [{}, {}, ..., {}]".format(i, i, i))
        initial_guess = list([[i]] * DEFAULT_MATRIX_SIZE)
        data.append(single_test(DEFAULT_MATRIX_SIZE, end_fun, method=method,
                                initial_guess=initial_guess))

    name = " od liczb w wektorze startowym z " \
           "warunkiem stopu {} dla metody {}" \
        .format(methods_inv(method), end_functions[end_fun])
    x_label = "Liczba w wektorze startowym"

    for i in range(3):
        plot(data, i, name, x_label, x_log=True, range_gen=range_gen)


def test_epsilon(end_fun: End_fun_sig, method: Method_signature):
    logger.info("epsilon test")
    data = []
    range_gen = list(2 ** -j for j in range(8, MAX_POW_EPS))
    for i in range_gen:
        logger.info("size: {}".format(i))
        data.append(single_test(DEFAULT_MATRIX_SIZE, end_fun,
                                epsilon=i, method=method))

    name = " od epsilonu z warunkiem stopu {} dla metody {}" \
        .format(methods_inv(method), end_functions[end_fun])
    x_label = "epsilon"

    for i in range(3):
        plot(data, i, name, x_label, x_log=True, range_gen=range_gen)


def test_SOR_factor(end_fun: End_fun_sig,
                    method: Method_signature = methods['SOR'],
                    interval: List[float] = None):
    if method != methods['SOR']:
        return

    logger.info("SOR factor")
    data = []

    if not interval:
        interval = list(float(j) / SOR_SCALE
                        for j in range(-0 * SOR_SCALE, 2 * SOR_SCALE + 1, 1)
                        if j != 0)
    for w in interval:
        logger.info("SOR parameter: {}".format(w))
        data.append(single_test(DEFAULT_MATRIX_SIZE, end_fun,
                                factor=w, method=method))

    name = " od parametru SOR z warunkiem stopu {} dla metody {}" \
        .format(methods_inv(method), end_functions[end_fun])
    x_label = "parameter metody SOR"

    for i in range(3):
        plot(data, i, name, x_label, range_gen=interval)


def experiment():
    for fun in end_functions.keys():
        for met in (methods['Jacobi'], methods['SOR']):
            for test in (test_vector, test_epsilon, test_size):
                test(fun, met)
        test_SOR_factor(fun, methods['SOR'])


def compare(matrix_size=100):
    a = []
    aa = []
    data1 = []
    data2 = []
    for i in range(1, matrix_size):
        w2 = single_test(i, end_function_2, method=methods['SOR'])
        epsilon = w2[0]
        w1 = single_test(i, end_function_1, method=methods['SOR'],
                         epsilon=epsilon)

        a.append(w1[0])
        aa.append(w2[0])
        data1.append(w1[3])
        data2.append(w2[3])

    plt.xlabel("Rozmaiar macierzy A")
    plt.ylabel("Czas obliczania [s]")
    plt.plot(range(1, matrix_size), data1, label=end_functions[end_function_1])
    plt.plot(range(1, matrix_size), data2, label=end_functions[end_function_2])
    plt.figure(num=1, figsize=(10, 20), dpi=100)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("data/" + "compare" + ".pdf")

    plt.clf()
    plt.xlabel("Rozmaiar macierzy A")
    plt.ylabel("Błąd względem poprawnego wyniku")
    plt.plot(range(1, matrix_size), a, label=end_functions[end_function_1])
    plt.plot(range(1, matrix_size), aa, label=end_functions[end_function_2])
    plt.figure(num=1, figsize=(10, 20), dpi=100)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("data/" + "compare2" + ".pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # experiment()

    compare()
