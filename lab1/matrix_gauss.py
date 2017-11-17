import lab1.precision as precision
from random import choice, seed
import matplotlib.pyplot as plt
import copy
import itertools
import logging
import sys

number = precision.number

logger = logging.getLogger('matrix.gauss')
logger.setLevel(logging.INFO)


def a1(i, j):
    if i == 1:
        return 1
    else:
        return number(1 / (number(number(i + j) - 1)))


def a2(i, j):
    return a2(j, i) if j < i else number(number(2 * i) / j)


def get_matrix(fun, n):
    assert n >= 1

    matrix = [None] * n
    for i in range(1, n+1):
        next_row = [None] * n
        for j in range(1, n+1):
            next_row[j-1] = fun(i, j)
        matrix[i-1] = next_row

    return matrix


def get_vertical_vector_x(n, random=lambda: choice((-1, 1))):
    assert n >= 1
    seed(7)
    return [[number(random())] for _ in range(n)]


def matrix_mul(matrix1, matrix2):
    assert len(matrix1[0]) == len(matrix2)

    i_len = len(matrix1)
    j_len = len(matrix2[0])
    k_len = len(matrix2)

    matrix = []
    for i in range(i_len):
        row = []
        for j in range(j_len):
            val = 0.
            for k in range(k_len):
                a = matrix1[i][k]
                b = matrix2[k][j]
                val += a * b
            row.append(val)
        matrix.append(row)

    return matrix


def gauss_elimination(matrix, vec):
    matrix_size = len(matrix)
    assert matrix_size == len(matrix[0])
    assert matrix_size == len(vec)

    matrix = copy.deepcopy(matrix)
    vec = [array[0] for array in vec]

    # convert to row echelon form
    for i in range(matrix_size-1):
        for ii in range(i+1, matrix_size):
            if matrix[ii][i] == 0:
                continue

            ratio = number(-matrix[ii][i] / matrix[i][i])

            # sub in matrix
            matrix[ii][i] = 0
            for j in range(i+1, matrix_size):
                matrix[ii][j] = number(matrix[ii][j]
                                       + number(ratio * matrix[i][j]))

            # sub in vector
            vec[ii] = number(vec[ii] + number(ratio * vec[i]))

    # calc vec values
    for i in range(matrix_size-1, -1, -1):
        div = matrix[i][i]
        vec[i] = number(vec[i] / div)

        # include vec[i] in all prev values
        for ii in range(i-1, -1, -1):
            vec[ii] = number(vec[ii] - number(matrix[ii][i] * vec[i]))
            matrix[ii][i] = number(0)

    return [[v] for v in vec]


def get_vector_distance(vec1, vec2):
    result = 0
    it1 = itertools.chain.from_iterable(vec1)
    it2 = itertools.chain.from_iterable(vec2)

    for a, b in zip(it1, it2):
        result = max(result, abs(a - b))

    return result


def single_test(fun, n):
    A = get_matrix(fun, n)
    x = get_vertical_vector_x(n)
    B = matrix_mul(A, x)

    X = gauss_elimination(A, B)
    distance = get_vector_distance(x, X)

    logger.debug("A: {}".format(A))
    logger.debug("x: {}".format(x))
    logger.debug("B: {}".format(B))
    logger.debug("X: {}".format(X))
    logger.info("maximum metric {}".format(distance))

    return distance


def experiment(fun, max_size=20):
    data = {}

    for float_name, float_type in precision.floating_point_types.items():
        global number
        number = float_type
        data[float_name] = []

        logger.info("precision type: {}".format(float_name))

        for n in range(1, max_size + 1):
            logger.info("matrix size {}".format(n))
            data[float_name].append(single_test(fun, n))

        logger.info("")

    plt.subplot(111)
    plt.semilogy()
    plt.title('Funkcja błędu od rozmiaru macierzy')
    plt.ylabel('Błąd w metryce maksimum')
    plt.xlabel('Rozmiar macierzy')
    for float_name in precision.floating_point_types.keys():
        plt.plot(range(1, max_size + 1), data[float_name], label=float_name)

    plt.legend(loc='lower right')
    plt.savefig('result.pdf')
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    print(sys.argv)

    if len(sys.argv) == 1:
        matrix_fun = a1
        m_size = 25
    else:
        matrix_fun = a2
        m_size = 100

    experiment(matrix_fun, m_size)
