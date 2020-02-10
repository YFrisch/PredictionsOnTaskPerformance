"""
    This script is used to read-in hand-drawn distributions about the confidence of a user about the points on a certain
    task
"""

# TODO: Check y values of discrete points; R.n. the need to get transformed by (1-...)

import numpy as np
import matplotlib.pyplot as plt


class DiscreteDistributionReader:

    def __init__(self, path, points):
        self.path = path
        self.points_per_task = points
        self.discrete_values = None
        self.indices = None
        self.ys = None
        self.xs = None
        # Read in image file at path
        self.img = plt.imread(self.path)
        # TODO: Transform the image into grayscale
        # self.img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        self.img_shape = np.shape(self.img)
        self.discretize()

    # Return discrete values per point from image
    def discretize(self):
        # TODO: Cut offsets from image?
        # TODO: Better threshold?
        #threshold = 0.9*(np.max(self.img) - np.min(self.img))
        #print("Threshold: ", threshold)
        threshold = 240
        xs = np.arange(0, self.img_shape[1])
        ys = []
        y_scale = self.img_shape[1]
        for x in xs:
            value = -1
            for y in np.arange(self.img_shape[0]-1, 0, -1):
                # print(self.img[y, x])
                if self.img[y, x] <= threshold:
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
        self.indices = np.arange(0, len(xs) + len(xs)/self.points_per_task, len(xs)/self.points_per_task, dtype=int)
        self.discrete_values = ys[self.indices]

        return None

    # Plot the discrete values for each point of the task
    def plot(self):
        # Creates two subplots and unpacks the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        f.suptitle(self.path)

        ax1.set_title("Original image")
        ax1.imshow(self.img, cmap="gray", origin='upper')

        # TODO: y-Axis should show 0.0 - 1.0 instead of 0.0 - max
        # TODO: x-Axis should show the discrete points rather than the pixel values
        ax2.scatter(self.xs, 1-self.ys, color='blue', alpha=0.1)
        ax2.scatter(self.indices, 1-self.discrete_values, color='red', s=120, alpha=1)
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            # TODO: Better alignment for annotations
            ax2.annotate(txt, (self.indices[i], 1-self.discrete_values[i]))
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Reached points")
        ax2.set_title("Discrete values")

        plt.show()
        return None

    # Return Brier-Score for discrete reached points
    def brier_score(self, points):
        ppt = np.zeros(shape=(self.points_per_task+1, 1))
        ppt[points] = 1
        bs = np.mean([((1-self.discrete_values[i]) - ppt[i])**2 for i in np.arange(0, len(ppt))])
        # print("Brier-Score of '{}' for {} points is {}.".format(self.path, points, bs))
        return bs

