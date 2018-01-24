import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
    """warunek: wspolczynnik: a * a * k / h / h < 0.5"""
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

    for time in range(0, time_div):
        for x in range(1, x_div):
            inner = arr[time, x + 1] - 2 * arr[time, x] + arr[time, x - 1]
            arr[time + 1, x] = coefficient * inner + arr[time, x]

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
