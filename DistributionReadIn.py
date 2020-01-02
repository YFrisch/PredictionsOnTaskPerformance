"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


class DistributionReader:

    def __init__(self):
        self.distributions = []

    # Return discrete function from image
    def fit_dist(self, image):

        # TODO: Cut offsets from image (x-axis

        xs = np.arange(0, np.shape(image)[1])
        ys = []
        for x in xs:
            value = -1
            for y in np.arange(np.shape(image)[0]-1, 0, -1):
                if image[y, x] <= 0.2:
                    value = y
                    break
            ys.append(value)
        ys = np.array(ys)

        # Cut data where y == -1 (no dark pixels)
        indices = np.where(ys == -1)[0]
        xs = np.delete(xs, indices)
        ys = np.delete(ys, indices)

        # Fit polynomial function to data
        poly_fit_deg = 2
        coeff = np.polyfit(xs, ys, deg=poly_fit_deg)

        # TODO: Discretize function

        # return Polynomial(coeff)
        return coeff

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
            x = np.arange(0, 250, 0.1)
            p = np.poly1d(self.distributions[index])
            y = np.array([p(xi) for xi in x])
            plt.plot(x, y)
            plt.title("Distribution {}".format(index))
            plt.show()
            return None

