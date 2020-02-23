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
        self.normalized_discrete_values = None
        self.indices = None
        self.ys = None
        self.xs = None
        # Read in image file at path
        self.img = plt.imread(self.path)
        # Greyscale transformation
        # self.img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        self.img_shape = np.shape(self.img)
        self.discretize()

    # Return discrete values per point from image
    def discretize(self):
        # TODO: Cut offsets from image?
        # TODO: Better threshold?
        threshold = 237
        print(self.img_shape)
        xs = np.arange(1, self.img_shape[1])
        ys = []
        y_scale = self.img_shape[0]
        for x in xs:
            value = -1
            for y in np.arange(0, self.img_shape[0], 1):
                if self.img[y, x] <= threshold:
                    value = y/y_scale
                    break
            ys.append(value)
        ys = np.array(ys)

        # Cut data where y == -1 (no dark pixels)
        step = 3
        for c in np.arange(0, len(ys)):
            c_low = np.clip(c, 0, None)
            c_up = np.clip(c, None, c + step)
            if ys[c] == -1:
                if not all(ys[np.arange(c_low, c_up+1, 1)]) == -1:
                    ys[c] = np.mean(ys[np.arange(c_low, c_up + 1, 1)])

        indices = np.where(ys == -1)[0]
        self.xs = np.delete(xs, indices)
        self.ys = np.delete(ys, indices)

        print("xs {} ys {}".format(xs.shape, ys.shape))

        # Indices of discrete points
        steps = np.floor(len(xs)/(self.points_per_task+1))
        print("Steps: ", steps)
        self.indices = np.arange(0+(steps/2), len(xs)-(steps/2), steps, dtype=int)
        print("Indices: ", self.indices)
        self.discrete_values = 1 - ys[self.indices]
        self.normalized_discrete_values = self.discrete_values/sum(self.discrete_values)

        return None

    # Plot the discrete values for each point of the task
    def plot(self):
        # Creates two subplots and unpacks the output array immediately
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        f.suptitle(self.path)

        ax1.set_title("Original image")
        ax1.imshow(self.img, cmap="gray", origin='upper')

        # TODO: y-Axis should show 0.0 - 1.0 instead of 0.0 - max
        # TODO: x-Axis should show the discrete points rather than the pixel values
        ax2.scatter(self.xs, 1-self.ys, color='blue', alpha=0.1)
        ax2.scatter(self.indices, self.discrete_values, color='red', s=120, alpha=1)
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            # TODO: Better alignment for annotations
            ax2.annotate(txt, (self.indices[i], self.discrete_values[i]))
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Reached points")
        ax2.set_title("Discrete values")

        ax3.scatter(self.indices, self.normalized_discrete_values, color='green', s=120, alpha=1)
        ax3.set_title("Normalized discrete values")
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            # TODO: Better alignment for annotations
            ax3.annotate(txt, (self.indices[i], self.normalized_discrete_values[i]))

        plt.show()
        return None

    # Return Brier-Score for discrete reached points
    def brier_score(self, points):
        ppt = np.zeros(shape=(self.points_per_task+1, 1))
        ppt[points] = 1
        bs = np.mean([((1-self.discrete_values[i]) - ppt[i])**2 for i in np.arange(0, len(ppt))])
        # print("Brier-Score of '{}' for {} points is {}.".format(self.path, points, bs))
        return bs

