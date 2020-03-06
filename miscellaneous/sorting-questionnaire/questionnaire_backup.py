import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math


def draw_random():
    nums = list(range(1, 6))
    random.shuffle(nums)
    print(nums)



def plot_example_probability_functions(with_points=False):

    def uni(x):
        return 10.0 / len(x)

    def expo(x):
        return 1 / np.exp(x)

    def weibull(x, alpha, beta):
        return alpha * beta * x ** (beta - 1) * np.exp(-alpha * x ** beta)

    def alpha(x, alpha):
        return alpha * np.exp(-7 * alpha * x)

    # Data
    x = np.arange(0, 1.015, 0.015)
    g_x = scipy.stats.norm.pdf(x, 0.3, math.sqrt(0.01))
    g2_x = scipy.stats.norm.pdf(x, 0.666, 0.3)
    u = uni(x)
    u_x = [u for xs in x]
    w_x = weibull(x, 1, 1)
    a_x = alpha(x, 1)

    # For plotting points
    points = np.arange(0, 1.1, 0.2)
    g_points = scipy.stats.norm.pdf(points, 0.3, math.sqrt(0.01))
    g2_points = scipy.stats.norm.pdf(points, 0.666, 0.3)
    a_points = alpha(points, 1)

    # Plots
    plt.figure()
    plt.plot(x, g_x, color=f'black')
    plt.ylim(0, 5)
    plt.yticks([0, 1, 2, 3, 4, 5], [0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.figure()
    plt.plot(x, g_x, color=f'black')
    plt.ylim(0, 5)
    plt.yticks([0, 1, 2, 3, 4, 5], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.scatter(points, g_points, color=f'magenta', marker=f'o', s=50)


    plt.figure()
    plt.plot(x, g2_x, color='black')

    plt.figure()
    plt.plot(x, g2_x, color='black')
    plt.scatter(points, g2_points, color=f'magenta', marker=f'o', s=50)


    plt.figure()
    plt.plot(x, a_x, color='black')

    plt.figure()
    plt.plot(x, a_x, color='black')
    plt.scatter(points, a_points, color=f'magenta', marker=f'o', s=50)

    plt.show()

plot_example_probability_functions(True)