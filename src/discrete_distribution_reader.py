"""This module is implementing a "discrete distribution reader" capable of reading drawn
probability density functions (pdfs) and storing their values.

The values are discretized for a fixed amount of points, and normalized afterwards to
fit to a discrete probability density function.

Plotting is implemented for a given task id.

Only arguments are the 4-letter subject (vpn) code and an array containing the subject's task scores.

https://www.statisticshowto.datasciencecentral.com/brier-score/
The "brier score" can be calculated for a given task id and actual points of the vpn.
"""


__author__ = 'Yannik P. Frisch, Maximilian A. Gehrke'
__date__ = '27-02-2020'


import numpy as np
import os
import cv2
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
        self.overall_confidence_img, self.o_img_shape = self.read_in_overall_confidence_image()
        print("# Read in drawn confidence distributions.")
        self.points_per_task = 5
        self.task_scores = task_scores
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
        """ This method reads in 8 files named 'pdf_task_i.jpg' from the vpn sub-folder
        and stores them inside self.confidence_images.

        :return: confidence_images, img_shapes: list of read-in image per task and shape per image
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

    def read_in_overall_confidence_image(self):
        """ This method reads in the file named 'pdf_task_1_to7.jpg' from the vpn sub-folder
        and stores it inside self.overall_confidence_image.

        :return: overall_confidence_image, o_img_shape: read-in overall confidence and shape of that image
        """
        img = plt.imread(BASE_DIR + "/assets/subjects/subject_" + self.vpn_code
                         + "/pdfs/pdf_task1to7.jpg")
        self.confidence_images.append(img)
        self.img_shapes.append(img.shape)
        # Greyscale transformation
        # img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        # Cropping image by 10 pixels each side
        img = img[10:-10, 10:-10]

        # Smoothening kernel
        k = np.ones((20, 20), np.float32)/2.5
        img = cv2.filter2D(img, -1, kernel=k)
        return img, img.shape

    def discretize(self):
        """ This function stores the pixel values ('continuous' pdf) from the images
        in self.confidence_images in self.xs and self.ys.

        Discrete values for every possible point of a task are stored in self.discrete_values
        respectively self.normalized_discrete_values.

        The indices from which the discrete pdf values are taken are also stored for every task
        in self.indices.
        :return: None
        """
        for i in range(0, len(self.confidence_images)):
            img = self.confidence_images[i]
            # Adaptive threshold for last image (overall confidence)
            if i < len(self.confidence_images) - 1:
                threshold = np.max(img) - (np.max(img)-np.min(img)) / 3.0
            else:
                threshold = 100
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

            """Extrapolating missing data.
            
            Read-in image pixels with y[x] = -1 (see above) are used for extrapolation.
            Starting at x, we iterate forward until we find an y[x'] != -1.  
            If such an x' exists, we assign all values between y[x] and y[x'] the value y[x'].
            If such an x' does not exist, we start again from x and iterate backwards.
            If the backward iteration does not find a valid value either, we assign y[x] = 1, so we get a uniform
            distribution after normalization.                
            """
            for c in np.arange(0, len(y_raw)):
                if y_raw[c] == -1:
                    c_up = c
                    up_successful = False
                    while c_up < len(y_raw)-1:
                        c_up += 1
                        if y_raw[c_up] != -1:
                            y_raw[c] = y_raw[c_up]
                            up_successful = True
                            break
                    if up_successful:
                        c = c_up
                        continue
                    else:
                        c_low = c
                        low_successful = False
                        while c_low > 0:
                            c_low -= 1
                            if y_raw[c_low] != -1:
                                y_raw[c] = y_raw[c_low]
                                low_successful = True
                                break
                        if low_successful:
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
        """Saves the discrete probability confidence values for each task to a csv file.

        :return: None
        """
        path = BASE_DIR + f'/assets/subjects/subject_{self.vpn_code}/analysis/{self.vpn_code}_probabilities.csv'
        panda_df = pd.DataFrame.from_dict(self.normalized_discrete_values, orient='index')
        panda_df.to_csv(path, sep=',')

    def save_brier_to_csv(self):
        """Saves the brier score of a subject for each task to a csv file.

        :return: None
        """
        # Read-in points per task
        path = BASE_DIR + f'/assets/subjects/subject_{self.vpn_code}/analysis/{self.vpn_code}_brier_scores.csv'
        brier_scores = [self.brier_score_for_task(task_id=t_id, vpn_points_for_task=points)
                        for t_id, points in enumerate(self.task_scores)]
        panda_df = pd.DataFrame(brier_scores,
                                columns=[f'Brier Score'],
                                index=[f'task_1', f'task_2', f'task_3', f'task_4',
                                       f'task_5', f'task_6', f'task_7', f'task_8'])
        panda_df.to_csv(path, sep=',')

    def brier_score_for_task(self, task_id, vpn_points_for_task):
        """Returns the brier score for this vpn for the given task id and the given amount of actual reached points.

        :return: bs, the Brier Score
        """
        ppt = np.zeros(shape=(self.points_per_task+1, 1))
        ppt[vpn_points_for_task] = 1
        bs = np.sum([((self.normalized_discrete_values[task_id][i]) - ppt[i]) ** 2 for i in np.arange(0, len(ppt))])
        return bs

    def plot(self, task_ids):
        """Plots the original image, discrete data and normalized pdf values for a given task id.
        The overall confidence can be plotted by maximum task id + 1

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
            ax2.set_ylim(bottom=0)
            for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
                ax2.annotate(txt, (self.indices[id][i] + 5, self.discrete_values[id][i] - 0.02))
            ax2.set_ylabel("Confidence")
            ax2.set_xlabel("Reached points")
            ax2.set_title("Discrete values")

            ax3.scatter(self.indices[id], self.normalized_discrete_values[id], color='green', s=120, alpha=1)
            ax3.set_title("Normalized discrete values")
            ax3.set_ylim(top=1)
            ax3.set_ylim(bottom=0)
            for i, txt in enumerate(np.arange(0, self.points_per_task+1)):
                ax3.annotate(txt, (self.indices[id][i] + 5, self.normalized_discrete_values[id][i] - 0.01))

            plt.show()
        return None

