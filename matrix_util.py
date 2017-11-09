from random import choice
import copy
import itertools
import logging

logger = logging.getLogger('jac.matrix')
logger.setLevel(logging.INFO)


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
    return [[random()] for _ in range(n)]


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

    for i in range(matrix_size-1):
        for ii in range(i+1, matrix_size):
            if matrix[ii][i] == 0:
                continue

            ratio = -matrix[ii][i] / matrix[i][i]

            # sub in matrix
            matrix[ii][i] = 0
            for j in range(i+1, matrix_size):
                matrix[ii][j] = matrix[ii][j] + ratio * matrix[i][j]

            # sub in vector
            if vec:
                vec[ii] = vec[ii] + ratio * vec[i]

    # calc vec values
    for i in range(matrix_size-1, -1, -1):
        div = matrix[i][i]
        vec[i] = vec[i] / div

        # include vec[i] in all prev values
        for ii in range(i-1, -1, -1):
            vec[ii] = vec[ii] - matrix[ii][i] * vec[i]
            matrix[ii][i] = 0

    return [[v] for v in vec]


def get_distance(vec1, vec2):
    result = 0
    it1 = itertools.chain.from_iterable(vec1)
    it2 = itertools.chain.from_iterable(vec2)

    for a, b in zip(it1, it2):
        result = max(result, abs(a - b))

    return result


def matrix_norm_2(matrix) -> float:
    ret = 0
    for i in range(1, len(matrix)):
        row = sum(val * val for val in matrix[i])
        ret = max(ret, row)
    return ret
