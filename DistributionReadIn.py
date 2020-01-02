"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


class DistributionReader:

    def __init__(self):
        self.distributions = np.array([0, "0", (0, 0)]).reshape((1, 3))

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
        plt.figure()
        plt.title(path)
        plt.imshow(img, cmap="gray")
        new_dist = np.array([self.fit_dist(img), path, np.shape(img)]).reshape((1, 3))
        self.distributions = np.concatenate((self.distributions, new_dist), axis=0)
        return None

    # Plot a distribution with spec file name
    def plot_dist(self, path):
        index = np.where(self.distributions[:, 1] == path)[0]
        if index.size == 0:
            print("Distribution with the given path name not found")
            return None
        else:
            index = index[0]
            plt.figure()
            x = np.arange(0, self.distributions[index][2][1], 0.1)
            p = np.poly1d(self.distributions[index][0])
            y = np.array([p(xi) for xi in x])
            plt.plot(x, y)
            plt.title("Fitted {}".format(self.distributions[index][1]))
            plt.show()
            return None

