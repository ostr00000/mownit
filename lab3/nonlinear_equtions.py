import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import itertools
import logging
from typing import List
from lab1.matrix_gauss import gauss_elimination as gauss

Mat = List[List]
EPSILON = 1e-7
logger = logging.getLogger(__name__)
MAX_ITER = 1000


def F_1(x, y, z):
    return [[x ** 2 + 2 * y ** 2 - z ** 2],
            [3 * x ** 2 - 4 * y ** 2 + 6 * z ** 2],
            [4 * x - y - z ** 2]]


def F_1_der(x, y, z):
    return [[2 * x, 4 * y, -2 * z],
            [6 * x, -8 * y, 12 * z],
            [4, -1, -2 * z]]


def F_2(x, y):
    return [[y - x / 2.], [2 * x * x - y * y - 7]]


def F_2_der(x, y):
    return [[-1. / 2, 1], [4 * x, -2 * y]]


def F(x, y, z) -> Mat:
    return [[x * x + y * y + z - 1],
            [2 * x * x - y * y - 4 * z * z + 3],
            [x * x + y + z - 1]]


def F_der(x, y, z) -> Mat:
    return [[2 * x, 2 * y, 1],
            [4 * x, -2 * y, -8 * z],
            [2 * x, 1, 1]]


def FF(x):
    return [[x ** 10 - (1 - x) ** 15]]


def FF_der(x):
    return [[10 * x ** 9 + 15 * (1 - x) ** 14]]


def A(x, y, z):
    return [[x ** 2 + y ** 2 - x ** 2 - 1],
            [x - 2 * y ** 3 + 2 * z ** 2 + 1],
            [2 * x ** 2 + y - 2 * z ** 2 - 1]]


def A_der(x, y, z):
    return [[2 * x, 2 * y, -2 * x],
            [x, -6 * y ** 2, 4 * z],
            [4 * x, y, -4 * z]]


# number of equations, system of equations, Jacobian
eq1 = (3, F_1, F_1_der)
eq2 = (2, F_2, F_2_der)
eq = (3, F, F_der)
eqq = (1, FF, FF_der)
eA = (3, A, A_der)


def unpack(matrix: Mat) -> List:
    return list(row[0] for row in matrix)


def get_distance_maximum_norm(a: Mat, b: Mat):
    result = 0
    it1 = itertools.chain.from_iterable(a)
    it2 = itertools.chain.from_iterable(b)

    for a, b in zip(it1, it2):
        result = max(result, abs(a - b))
    return result


def end_function_1(old_vec: Mat, new_vector: Mat,
                   _fun=None, epsilon: float = EPSILON):
    return get_distance_maximum_norm(old_vec, new_vector) < epsilon


def end_function_2(_old_vec: Mat, new_vector: Mat,
                   fun=None, epsilon: float = EPSILON):
    max_val = max(row[0] for row in fun(*unpack(new_vector)))
    return max_val < epsilon


def sub_matrix(matrix_a: Mat, matrix_b: Mat) -> Mat:
    assert len(matrix_a) == len(matrix_b)
    assert len(matrix_a[0]) == len(matrix_b[0])

    result = []
    for a, b in zip(matrix_a, matrix_b):
        row = [aa - bb for aa, bb in zip(a, b)]
        result.append(row)
    return result


def newton_method(end_function, equations_size=3, fun=F, fun_der=F_der,
                  start_vec: Mat = None):
    start_vec = start_vec or [[1]] * equations_size

    old_vec = start_vec
    iterations = 0
    while True:
        iterations += 1

        values_vector = fun(*unpack(old_vec))
        jacobi_matrix = fun_der(*unpack(old_vec))
        try:
            x_vec = gauss(jacobi_matrix, values_vector)
            # x_vec = np.linalg.solve(np.array(jacobi_matrix),
            #                         np.array(values_vector)).tolist()
        except:
            return [], MAX_ITER + 1

        new_vec = sub_matrix(old_vec, x_vec)

        if end_function(old_vec, new_vec, fun) or iterations > MAX_ITER:
            return new_vec, iterations

        old_vec = new_vec


def is_near(real):
    def check(val):
        total = 0
        for r, v in zip(real, val):
            total = max(total, abs(v[0] - r))
        return total < 0.1

    return check


def make_single_measure(vec, data):
    result, it = newton_method(end_function_1, start_vec=vec)
    # result, it = newton_method(end_function_2, start_vec=vec)

    if it <= MAX_ITER:
        for cont, fun in data.values():
            if fun(result):
                cont.append(unpack(vec))
                break
        else:
            logger.warning("unknown point")
        logger.debug("for (x,y,z): ({}, {}, {}) result is: {}"
                     .format(*unpack(vec), result))
    elif result:
        logger.warning("max iteration {}".format(vec))
    else:
        logger.warning("Singular Matrix")


def make_measurement(start, end, num, data):
    for x, y, z in itertools.product(np.linspace(start, end, num), repeat=3):
        logger.debug("test x: {}, y: {}, z: {}".format(x, y, z))
        vector = [[x], [y], [z]]
        make_single_measure(vector, data)


def find():
    data = {
        'r': ([], is_near((-1, 1, -1))),
        'b': ([], is_near((1, 1, -1))),
        'g': ([], is_near((-0.323, 0, 0.896))),
        'c': ([], is_near((0.323, 0, 0.896))),
        'k': ([], is_near((-1.548, 0, -1.396))),
        'm': ([], is_near((1.548, 0, -1.396)))

    }
    #
    # Arek_data = {
    #     'r': ([], is_near((-1, 1, -1))),
    #     'b': ([], is_near((-1, 1, 1))),
    #     'g': ([], is_near((0.5, 1, -0.5))),
    #     'k': ([], is_near((0.5, 1, 0.5)))
    # }

    for i in range(7, 13):
        make_measurement(-10, 10, i, data)
    # make_measurement(-10, 10, 7, data)

    # for x in np.linspace(-10, 10, 6):
    #     for y in np.linspace(0, 10, 6):
    #         for z in np.linspace(0, 10, 6):
    #             make_single_measure([[x], [y], [z]], data)

    fig = plt.figure(figsize=(10, 20), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    for symbol, (points, _) in data.items():
        if points:
            ax.scatter(*zip(*points), c=symbol, marker='o', s=100)

    ax.set_xlabel('oś X ')
    ax.set_ylabel('oś Y')
    ax.set_zlabel('oś Z')

    plt.show()
    fig.savefig("data/system_eq.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler('results.txt', mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    find()
    # print(newton_method(end_function_1, *eA, start_vec=[[2], [2], [2]]))
