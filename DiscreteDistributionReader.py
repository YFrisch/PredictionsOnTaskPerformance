"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

import numpy as np
import matplotlib.pyplot as plt


class DiscreteDistributionReader:

    def __init__(self, path, points):
        self.path = path
        self.points_per_task = points
        self.discrete_values = None
        self.img_shape = None
        self.img = None
        self.indices = None
        self.ys = None
        self.xs = None
        self.read_in_dist(self.path, self.points_per_task)


    # Return discrete values per point from image
    def discretize(self):

        image = self.img

        # TODO: Cut offsets from image

        threshold = 0.5*(np.max(image) + np.min(image))

        xs = np.arange(0, np.shape(image)[1])
        ys = []
        y_scale = np.shape(image)[1]
        for x in xs:
            value = -1
            for y in np.arange(np.shape(image)[0]-1, 0, -1):
                if image[y, x] <= threshold:
                    value = y/y_scale
                    break
            ys.append(value)
        ys = np.array(ys)

        # Cut data where y == -1 (no dark pixels)
        indices = np.where(ys == -1)[0]
        xs = np.delete(xs, indices)
        ys = np.delete(ys, indices)
        self.xs = xs
        self.ys = ys

        # Indices of discrete points
        #TODO: Better step size ? RN we use the middle of the point "intervalls" for every discrete x value
        self.indices = [int((i+0.5)*len(xs)/(self.points_per_task+1)) for i in range(0, self.points_per_task+1)]
        self.discrete_values = ys[self.indices]

        return None

    # Read image, transform it to grayscale, get discrete points and append them to itern list
    def read_in_dist(self, path, points):
        img = plt.imread(path)
        img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        self.img_shape = np.shape(img)
        self.img = img

        self.points_per_task = points

        self.discretize()


        return None

    # Plot the discrete values for each point of the task
    def plot(self):
        # Creates two subplots and unpacks the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        f.suptitle(self.path)

        ax1.set_title("Original image")
        ax1.imshow(self.img, cmap="gray", origin='lower')

        # TODO: x-Axis should show the discrete points rather than the pixel values
        #plt.scatter(np.arange(0, self.points_per_task+1), self.discrete_values)
        ax2.scatter(self.xs, self.ys, color='blue', alpha=0.1)
        ax2.scatter(self.indices, self.discrete_values, color='red', s=120, alpha=1)
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            # TODO: Better alignment for annotations
            ax2.annotate(txt, (self.indices[i], self.discrete_values[i]))
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Reached points")
        ax2.set_title("Discrete values")

        plt.show()
        return None
