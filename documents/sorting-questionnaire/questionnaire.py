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


def plot_example_probability_functions_2():
    # functions
    def gaussian(x, mu, sigma):
        return (1.0 / (sigma * 2 * np.sqrt(np.pi))) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2)

    def uni(x):
        return 10.0 / len(x)

    def expo(x):
        return 1 / np.exp(x)

    def weibull(x, alpha, beta):
        return alpha * beta * x ** (beta - 1) * np.exp(-alpha * x ** beta)

    def alpha(x, alpha):
        return alpha * np.exp(-7 * alpha * x)

    # Data
    x = np.arange(0, 1, 0.015)
    g_x = gaussian(x, mu=0.666, sigma=0.3)
    u = uni(x)
    u_x = [u for xs in x]
    w_x = weibull(x, 1, 1)
    a_x = alpha(x, 1)

    # Plots
    plt.figure()
    plt.plot(x, g_x, color='black')
    plt.ylim(0, 1)

    plt.figure()
    plt.plot(x, u_x, color='black')
    plt.ylim(0, 1)

    plt.figure()
    plt.plot(x, a_x, color='black')

    plt.show()

# plot_example_probability_functions(True)



def plot_brier_graph():
    """
    This function draws a probability density function.

    The peak of the function shall be at 70% and the right and left event
    shall have 15%. The other events shall have 0%. This plot shall
    illustrate how well the participants did in our experiment if we
    transform the Brier score into a probability density function
    """

    # Data
    x = np.arange(0, 1.015, 0.015)
    mean = 0.4
    sd = math.sqrt(0.013)
    g_x = scipy.stats.norm.pdf(x, mean, sd)

    # Check if 0.2 and 0.6 have 0.15 and 0.4 has 0.7
    scatter_x = [0.2, 0.4, 0.6]
    scatter_y = []
    for xi in scatter_x:
        y = scipy.stats.norm.pdf(xi, mean, sd)
        scatter_y.append(y)
        print(f'({xi}, {y})')

    # Plots
    plt.figure()
    plt.plot(x, g_x, color=f'black')
    plt.ylim(0, 5)
    plt.yticks([0, 1, 2, 3, 4, 5], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 1, 2, 3, 4, 5])
    plt.xlabel(f'Points')
    plt.ylabel(f'Probability')
    plt.fill_between(x, g_x, alpha=.9)
    plt.show()







plot_brier_graph()