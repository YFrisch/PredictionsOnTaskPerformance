"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

import numpy as np
import matplotlib.pyplot as plt


# Class for polynomial distributions
class Polynomial:

    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __repr__(self):
        return "Polynomial" + str(self.coefficients)

    def __call__(self, x):
        res = 0
        for index in np.arange(len(self.coefficients)-1, -1, -1):
            res += self.coefficients[index] * x ** index
        return res


class DistributionReader:

    def __init__(self):
        self.distributions = []

    # Return discrete function from image
    def fit_dist(self, image):
        # TODO: Get offsets for x and y
        x_offsets = (0, 0)
        y_offsets = (0, 0)
        # TODO: Cut offsets from image
        # TODO: Get values from pixels

        xs = np.arange(0, np.shape(image)[0])
        ys = []
        for x in xs:
            value = 0
            for y in np.arange(0, np.shape(image)[1]):
                if image[x, y] <= 0.1:
                    value = y
                    break
            ys.append(value)

        plt.figure()
        plt.imshow(image)
        plt.show()

        plt.figure()
        plt.scatter(xs, ys)
        plt.show()

        # Fit polynomial function to data
        poly_fit_deg = 3
        coeff = np.polyfit(xs, ys, deg=poly_fit_deg)
        # TODO: Discretize function
        return Polynomial(coeff)

    # Read image, transform it to grayscale, get distribution and append to distribution list
    def read_in_dist(self, path):
        img = plt.imread(path)
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        self.distributions.append(self.fit_dist(img))
        return None

    # Plot a distribution with spec index
    def plot_dist(self, index):
        if index >= len(self.distributions):
            print("Invalid distribution index")
            return None
        else:
            plt.figure()
            x = np.arange(0, 10, 0.1)
            y = np.array([self.distributions[index](xi) for xi in x])
            plt.plot(x, y)
            plt.title("Distribution {}".format(index))
            plt.show()
            return None

