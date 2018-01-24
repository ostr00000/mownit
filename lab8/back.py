import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.linalg import solve_banded

left = 0
right = np.pi

a = 0.5
t_c = 5
A = 1
alpha = 2
B = 0


def phi(x):
    return A * np.sin(alpha * x)


def psi_left(time):
    return 0


def psi_right(time):
    return B



def differential_equation(time_div, x_div):
    arr = np.zeros(shape=(time_div + 1, x_div + 1))

    x_values = np.linspace(left, right, x_div + 1)
    t_values = np.linspace(0, t_c, time_div + 1)

    for t_index, t_value in enumerate(t_values):
        arr[t_index, 0] = psi_left(t_value)
        arr[t_index, x_div] = psi_right(t_value)

    for x_index, x_value in enumerate(x_values):
        arr[0, x_index] = phi(x_value)

    h = (right - left) / x_div
    k = t_c / time_div
    coefficient = a * a * k / h / h
    print("wspolczynnik = {}".format(coefficient))

    coef = h * h / a * a / k
    middle_coef = 2 + coef

    assert x_div > 3
    top, mid, bot = [0, 0], [1, middle_coef], [-1, -1]
    for t in range(2, x_div - 1):
        top.append(-1)
        mid.append(middle_coef)
        bot.append(-1)

    top.extend([-1, -1])
    mid.extend([middle_coef, 1])
    bot.extend([0, 0])

    band_matrix = np.array([top, mid, bot])

    for t in range(1, time_div + 1):
        vector = np.zeros(x_div + 1)

        vector[0] = arr[t, 0]
        vector[x_div] = arr[t, x_div]

        for i in range(1, x_div):
            vector[i] = arr[t - 1, i] * coef

        result = solve_banded((1, 1), band_matrix, vector)
        arr[t] = result
        print(result)

    return [x_values, t_values, arr]


def draw(data):
    x, y = np.meshgrid(data[0], data[1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, data[2])
    ax.set_xlabel("oś X")
    ax.set_ylabel("czas")
    ax.set_zlabel("wartość")

    plt.show()


if __name__ == '__main__':
    dif_eq_data = differential_equation(500, 40)
    draw(dif_eq_data)
