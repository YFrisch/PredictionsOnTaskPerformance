""" This module is implementing a "discrete distribution reader" capable of reading drawn
    probability density functions (pdfs) and storing their values.

    The values are discretized for a fixed amount of points, and normalized afterwards to
    fit to a discrete probability density function.

    Plotting is implemented for a given task id.

    Only argument is a 4-letter subject (vpn) code.

    # TODO: Add source
    The "brier score" can be calculated for a given task id and actual points of the vpn.
"""

__author__ = 'Yannik Frisch'
__date__ = '27-02-2020'

import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


class DiscreteDistributionReader:

    def __init__(self, vpn_code):
        self.vpn_code = vpn_code
        print("\n------------------------------")
        print("# Created distribution reader for vpn {}.".format(vpn_code))
        self.confidence_images, self.img_shapes = self.read_in_confidence_images()
        print("# Read in drawn confidence distributions.")
        self.points_per_task = 5
        self.indices = defaultdict(np.ndarray)
        self.ys = defaultdict(np.ndarray)
        self.xs = defaultdict(np.ndarray)
        self.discrete_values = defaultdict(np.ndarray)
        self.normalized_discrete_values = defaultdict(np.ndarray)
        self.discretize()
        print("# Discretized distributions.")
        print("------------------------------")

    def read_in_confidence_images(self):
        """ This method reads in 8 files named 'pdf_task_i.jpg' from the vpn subfolder
            and stores the inside self.confidence_images.
        """
        confidence_images = []
        img_shapes = []
        for i in range(1, 9):
            img = plt.imread(BASE_DIR + "/assets/subjects/subject_" + self.vpn_code
                             + "/pdfs/pdf_task_" + str(i) + ".jpg")
            # Greyscale transformation
            # img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            confidence_images.append(img)
            img_shapes.append(np.shape(img))
        return confidence_images, img_shapes

    def discretize(self):
        """ This function stores the pixel values ('continous' pdf) from the images
            in self.confidence_images in self.xs and self.ys.

            Discrete values for every possible point of a task are stored in self.discrete_values
            respectively self.normalized_discrete_values.

            The indices from which the discrete pdf values are taken are also stored for every task
            in self.indices.
        """
        for i in range(0, len(self.confidence_images)):
            img = self.confidence_images[i]
            # TODO: Cut offsets from image?
            # TODO: Better threshold? Global threshold for img or local for each column?
            # threshold = 237
            threshold = np.max(img) - (np.max(img)-np.min(img))/5.0
            x_raw = np.arange(1, self.img_shapes[i][1])
            y_raw = []
            y_scale = self.img_shapes[i][0]
            for xc in np.arange(0, len(x_raw)):
                value = -1
                # threshold = np.max(img[:, x_raw[xc]]) - (np.max(img[:, x_raw[xc]]) - np.min(img[:, x_raw[xc]])) / 5.0
                for y in np.arange(0, self.img_shapes[i][0], 1):
                    if img[y, x_raw[xc]] <= threshold:
                        value = y/y_scale
                        break
                y_raw.append(value)
            y_raw = np.array(y_raw)

            # TODO: Extrapolate missing data; Not working yet
            step = 1
            for c in np.arange(0, len(y_raw)):
                c_low = np.clip(c - step, 0, None)
                c_up = np.clip(c + step, None, len(y_raw)-1)
                if y_raw[c] == -1:
                    current_pixels = np.copy(y_raw[np.arange(c_low, c_up+1, 1)])
                    if not all(current_pixels == -1) or not all(current_pixels == 0):
                        # Use average of not -1 values for extrapolation
                        values_for_expol = np.delete(current_pixels, np.where(current_pixels == -1)[0])
                        y_raw[c] = np.mean(values_for_expol)
                    else:
                        y_raw[c] = 1

            # Cut data where y == -1 (no dark pixels)
            indices = np.where(y_raw == -1)[0]
            self.xs[i] = np.delete(x_raw, indices)
            self.ys[i] = np.delete(y_raw, indices)

            # Indices of discrete points
            self.indices[i] = np.linspace(10, len(self.xs[i])-10, self.points_per_task+1).astype('i8')
            self.discrete_values[i] = 1 - self.ys[i][self.indices[i]]
            self.normalized_discrete_values[i] = self.discrete_values[i]/sum(self.discrete_values[i])

        return None

    # Plot the discrete values for each point of the task
    def plot(self, task_id):
        task_id = task_id - 1
        """ Plots the original image, discrete data and normalized pdf values for a given task id.
        """
        # Creates two subplots and unpacks the output array immediately
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
        f.suptitle(self.vpn_code + ": Task " + str(task_id))
        ax1.set_title("Original image")
        ax1.imshow(self.confidence_images[task_id], cmap="gray", origin='upper')

        # TODO: x-Axis should show the discrete points (0-1) rather than the pixel values
        ax2.scatter(self.xs[task_id], 1-self.ys[task_id], color='blue', alpha=0.1)
        ax2.scatter(self.indices[task_id], self.discrete_values[task_id], color='red', s=120, alpha=1)
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            ax2.annotate(txt, (self.indices[task_id][i]+5, self.discrete_values[task_id][i]-0.05))
        ax2.set_ylabel("Confidence")
        ax2.set_xlabel("Reached points")
        ax2.set_title("Discrete values")

        ax3.scatter(self.indices[task_id], self.normalized_discrete_values[task_id], color='green', s=120, alpha=1)
        ax3.set_title("Normalized discrete values")
        for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
            ax3.annotate(txt, (self.indices[task_id][i]+5, self.normalized_discrete_values[task_id][i]-0.01))

        plt.show()
        return None

    # Return Brier-Score for discrete reached points
    def brier_score(self, task_id, vpn_points_for_task):
        """ Returns the brier score for this vpn for the given task id and the given amount of actual reached points.
        """
        ppt = np.zeros(shape=(self.points_per_task+1, 1))
        ppt[vpn_points_for_task] = 1
        bs = np.mean([((1-self.discrete_values[task_id][i]) - ppt[i])**2 for i in np.arange(0, len(ppt))])
        print("\nBrier-Score of '{}' for {} points is {}.".format(self.vpn_code, vpn_points_for_task, np.round(bs, 3)))
        return bs

