"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

import numpy as np
import matplotlib.pyplot as plt


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
                if image[x, y] == 0:
                    value = y
                    break
            ys.append(value)

        # TODO: Fit function to image values
        coeff = np.polyfit(xs, ys, deg=3)
        # TODO: Discretize function
        return (coeff, 3)

    # Read image, transform it to grayscale, get distribution and append to distribution list
    def read_dist(self, path):
        img = plt.imread(path)
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        self.distributions.append(self.fit_dist(img))
        return None

    # Return all read-in distributions
    def return_dist(self):
        return self.distributions

    # Plot a distribution with spec index
    def plot_dist(self, index):
        if index >= len(self.distributions):
            print("Invalid distribution index")
            return None
        else:
            plt.figure()
            x = np.arange(0, 10)
            y = self.distributions[index](x)
            plt.plot(x, y)
            plt.title("Distribution {}".format(index))
            plt.show()
            return None

