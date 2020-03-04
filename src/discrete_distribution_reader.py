"""
    This module is implementing a "discrete distribution reader" capable of reading drawn
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
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


class DiscreteDistributionReader:

    def __init__(self, vpn_code, task_scores):
        self.vpn_code = vpn_code
        print("\n------------------------------")
        print("# Created distribution reader for vpn {}.".format(vpn_code))
        self.confidence_images, self.img_shapes = self.read_in_confidence_images()
        print("# Read in drawn confidence distributions.")
        self.points_per_task = 5
        self.task_scores =task_scores
        self.indices = defaultdict(np.ndarray)
        self.ys = defaultdict(np.ndarray)
        self.xs = defaultdict(np.ndarray)
        self.discrete_values = defaultdict(np.ndarray)
        self.normalized_discrete_values = defaultdict(np.ndarray)
        self.discretize()
        print("# Discretized distributions.")
        self.save_prob_to_csv()
        print("# Saved discrete values to .csv file.")
        self.save_brier_to_csv()
        print("# Saved brier scores to .csv file.")
        print("------------------------------")

    def read_in_confidence_images(self):
        """ This method reads in 8 files named 'pdf_task_i.jpg' from the vpn subfolder
            and stores the inside self.confidence_images.
        :returnn: confidence_images, img_shapes: list of read-in image per task and shape per image
        """
        confidence_images = []
        img_shapes = []
        for i in range(1, 9):
            img = plt.imread(BASE_DIR + "/assets/subjects/subject_" + self.vpn_code
                             + "/pdfs/pdf_task_" + str(i) + ".jpg")
            # Greyscale transformation
            # img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            # Cropping image by 5 pixels each side
            img = img[5:-5, 5:-5]
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
        :return: None
        """
        for i in range(0, len(self.confidence_images)):
            img = self.confidence_images[i]
            # TODO: Cut offsets from image?
            # TODO: Better threshold? Global threshold for img or local for each column?
            # threshold = 237
            threshold = np.max(img) - (np.max(img)-np.min(img))/3.0
            x_raw = np.arange(1, self.img_shapes[i][1])
            y_raw = []
            y_scale = self.img_shapes[i][0]
            for xc in np.arange(0, len(x_raw)):
                value = -1
                for y in np.arange(0, self.img_shapes[i][0], 1):
                    if img[y, x_raw[xc]] <= threshold:
                        value = y/y_scale
                        break
                y_raw.append(value)
            y_raw = np.array(y_raw)

            # Extrapolating missing data
            # TODO: Documentation / Comments
            for c in np.arange(0, len(y_raw)):
                if y_raw[c] == -1:
                    c_up = c
                    up_succesfull = False
                    while c_up < len(y_raw)-1:
                        c_up += 1
                        if y_raw[c_up] != -1:
                            y_raw[c] = y_raw[c_up]
                            up_succesfull = True
                            break
                    if up_succesfull:
                        c = c_up
                        continue
                    else:
                        c_low = c
                        low_succesfull = False
                        while c_low > 0:
                            c_low -= 1
                            if y_raw[c_low] != -1:
                                y_raw[c] = y_raw[c_low]
                                low_succesfull = True
                                break
                        if low_succesfull:
                            y_raw[c:] = y_raw[c]
                            break
                else:
                    continue

            # Cut data where y == -1 (no dark pixels)
            indices = np.where(y_raw == -1)[0]
            self.xs[i] = np.delete(x_raw, indices)
            self.ys[i] = np.delete(y_raw, indices)

            # Indices of discrete points
            self.indices[i] = np.linspace(10, len(self.xs[i])-10, self.points_per_task+1).astype('i8')
            self.discrete_values[i] = 1 - self.ys[i][self.indices[i]]
            self.normalized_discrete_values[i] = self.discrete_values[i]/sum(self.discrete_values[i])

        return None

    def save_prob_to_csv(self):
        """ Saves the discrete probability confidence values for each task to a csv file.
        :return: None
        """
        path = BASE_DIR + f'/assets/subjects/subject_{self.vpn_code}/analysis/{self.vpn_code}_probabilities.csv'
        panda_df = pd.DataFrame.from_dict(self.normalized_discrete_values, orient='index')
        panda_df.to_csv(path, sep=',')

    def save_brier_to_csv(self):
        """ Saves the brier score of a subject for each task to a csv file.
        :return: None
        """
        # Read-in points per task
        path = BASE_DIR + f'/assets/subjects/subject_{self.vpn_code}/analysis/{self.vpn_code}_brier_scores.csv'
        panda_df = pd.DataFrame([self.brier_score_for_task(task_id=id, vpn_points_for_task=score) for
                                           id, score in enumerate(self.task_scores)])
        panda_df.to_csv(path, sep=',')

    def brier_score_for_task(self, task_id, vpn_points_for_task):
        """ Returns the brier score for this vpn for the given task id and the given amount of actual reached points.
        :return: bs, the Brier Score
        """
        ppt = np.zeros(shape=(self.points_per_task+1, 1))
        ppt[vpn_points_for_task] = 1
        bs = np.mean([((1-self.discrete_values[task_id][i]) - ppt[i])**2 for i in np.arange(0, len(ppt))])
        print("\nBrier-Score of '{}' for {} points is {}.".format(self.vpn_code, vpn_points_for_task, np.round(bs, 3)))
        return bs

    def plot(self, task_ids):
        """ Plots the original image, discrete data and normalized pdf values for a given task id.
        :return: None
        """
        for id in task_ids:
            id = id - 1

            # Creates two subplots and unpacks the output array immediately
            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
            f.suptitle(self.vpn_code + ": Task " + str(id + 1))
            ax1.set_title("Original image")
            ax1.imshow(self.confidence_images[id], cmap="gray", origin='upper')

            ax2.scatter(self.xs[id], 1 - self.ys[id], color='blue', alpha=0.1)
            ax2.scatter(self.indices[id], self.discrete_values[id], color='red', s=120, alpha=1)
            ax2.set_ylim(top=1)
            for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
                ax2.annotate(txt, (self.indices[id][i] + 5, self.discrete_values[id][i] - 0.02))
            ax2.set_ylabel("Confidence")
            ax2.set_xlabel("Reached points")
            ax2.set_title("Discrete values")

            ax3.scatter(self.indices[id], self.normalized_discrete_values[id], color='green', s=120, alpha=1)
            ax3.set_title("Normalized discrete values")
            ax3.set_ylim(top=1)
            for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
                ax3.annotate(txt, (self.indices[id][i] + 5, self.normalized_discrete_values[id][i] - 0.01))

            plt.show()
        return None

