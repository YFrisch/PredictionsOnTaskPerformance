""" This file is used for several evaluations and plots of the overall assigned probabilities of the subjects.

    All available evaluation data is read-in first.

    Per default by calling this script, calibrate() is evaluated on all available subjects in the subject folder.
"""
from collections import OrderedDict, defaultdict
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# TODO: Get rid of the RGBA warning...
from matplotlib.pyplot import cm
import src.utils

__author__ = 'Yannik Frisch'
__date__ = '08-03-2020'

# Set working directory to the top level of our project
src.utils.set_working_directory()

print("\n-------------------- CALIBRATION EVALUATION --------------------")

# --------------- READ DATA --------------- #
print(f'# Reading data ... ', end='')
sys.stdout.flush()
subjects_folder_path = f'assets/subjects/'
subjects = src.utils.extract_subject_codes_from_folders(subjects_folder_path)

# Create plots folder if it is not there
src.utils.create_folder(f'assets/plots/')

subject_brier_scores = {}
subject_probs = {}
subject_task_scores = {}

for subject in subjects:
    # Read brier scores
    path_to_csv = f'assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_brier_scores.csv'
    pandas_frame = pd.read_csv(path_to_csv, sep=',')
    pandas_frame = pandas_frame.drop([7])  # Drop task 8
    pandas_bs = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
    subject_brier_scores[subject] = pandas_bs

    # Read probabilites
    path_to_csv = f'assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_probabilities.csv'
    pandas_frame = pd.read_csv(path_to_csv, sep=',')
    pandas_frame = pandas_frame.drop([7, 8])
    probs = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
    subject_probs[subject] = probs

    # Read task scores
    path_to_csv = f'assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_task_scores.csv'
    pandas_frame = pd.read_csv(path_to_csv, sep=',')
    pandas_frame = pandas_frame.drop([7])  # Drop task 8
    scores = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
    subject_task_scores[subject] = scores

print(f'Done!')


# --------------- EVALUATE DATA --------------- #
def calibration(subject_code):
    # Gather data
    ts = np.array(list(subject_task_scores.get(subject_code).values())).T.squeeze().reshape(-1, 1)
    prob_matrix = np.empty((1, 6))
    prob_dict = subject_probs.get(subject_code)
    for i in range(0, 7):
        task_prob = np.array(prob_dict.get(i)).squeeze().reshape((1, -1))
        prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
    prob_matrix = np.copy(prob_matrix[1:, :])

    # Get the peak of every discrete confidence distribution per task
    # TODO: Use expectation instead of max/argmax
    subject_predictions = np.argmax(prob_matrix, axis=1).reshape(-1, 1)/6

    # Plot achieved percentage of points vs predictions
    plt.figure(figsize=(10, 4))
    plt.title(f"Calibration of {subject_code}")
    color = plt.get_cmap('rainbow', len(ts))
    colors = color(np.linspace(0, 1, len(ts)))
    for i in range(0, len(ts)):
        # Todo: Make max task score modular
        max_score = 5

        task_color = colors[i].reshape(1, -1).squeeze()

        a = np.array([subject_predictions[i], ts[i]/max_score]).reshape(2,)
        plt.scatter(a[0], a[1], s=100, label=f'Task {i+1}', c=task_color, zorder=2)
        b = np.array([0.5, 0.5]).reshape(2,)
        correction = np.dot(a, b)/np.linalg.norm(b, 2)
        correction = correction*b

        """
        if subject_predictions[i] > ts[i]/max_score:
            plt.plot([a[0], correction[0]], [a[1], correction[1]],
                     color='red', label='Overconfidence', alpha=0.7)
        else:
            plt.plot([a[0], correction[0]], [a[1], correction[1]],
                     color='blue', label='Underconfidence', alpha=0.7)
        """

    plt.plot([0, 1], [0, 1], color='black', label='Optimal accuracy', ls='-', zorder=1)
    plt.xlim(left=0, right=1)
    plt.xlabel("Subjects predicted accuracy")
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Subjects actual accuracy")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def average_calibration():
    max_score = 5
    task_score_matrix = np.zeros(shape=(1, 7))
    subjective_prediction_matrix = np.zeros(shape=(1, 7))
    for s in subjects:
        # Gather data
        new_score = np.array(list(subject_task_scores.get(s).values())).reshape(1, -1)
        task_score_matrix = np.concatenate((task_score_matrix, new_score), axis=0)
        prob_matrix = np.empty((1, 6))
        prob_dict = subject_probs.get(s)
        for i in range(0, 7):
            task_prob = np.array(prob_dict.get(i)).squeeze().reshape((1, -1))
            prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
        prob_matrix = np.copy(prob_matrix[1:, :])
        new_probs = np.max(prob_matrix, axis=1).reshape(1, -1)
        # Get the peak of every discrete confidence distribution per task
        subjective_prediction_matrix = np.concatenate((subjective_prediction_matrix, new_probs), axis=0)

    # Delete first indices
    task_score_matrix = np.copy(task_score_matrix[1:, :])
    subjective_prediction_matrix = np.copy(subjective_prediction_matrix[1:, :])

    # Get averages of values (7 tasks -> 7 estimated accuracies and 7 actual accuracies
    average_actual_accuracy = np.mean(task_score_matrix, axis=0)/max_score
    actual_std = np.std(task_score_matrix, axis=0)/max_score
    average_predicted_accuray = np.mean(subjective_prediction_matrix, axis=0)
    prediction_std = np.std(subjective_prediction_matrix, axis=0)

    # Plot
    plt.figure(figsize=(10, 4))
    color = plt.get_cmap('rainbow', len(average_predicted_accuray))
    colors = color(np.linspace(0, 1, len(average_predicted_accuray)))
    for i in range(0, len(average_predicted_accuray)):
        task_color = colors[i].reshape(1, -1).squeeze()
        plt.scatter(average_predicted_accuray[i], average_actual_accuracy[i],
                    label=f'Task {i+1}', zorder=5, c=task_color)
        plt.plot([average_predicted_accuray[i] - prediction_std, average_predicted_accuray[i] + prediction_std],
                 [average_actual_accuracy[i], average_actual_accuracy[i]],
                 ls='dotted', alpha=0.4, zorder=1, c=task_color)
        plt.plot([average_predicted_accuray[i], average_predicted_accuray[i]],
                 [average_actual_accuracy[i] - actual_std, average_actual_accuracy[i] + actual_std],
                 ls='dotted', alpha=0.4, zorder=1, c=task_color)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.plot([0, 1], [0, 1], color='black', label='Optimal accuracy', ls='-', zorder=2)
    plt.legend(by_label.values(), by_label.keys())
    plt.title('Average calibration')
    plt.xlim(left=0, right=1)
    plt.xlabel("Average predicted accuracy")
    plt.ylim(bottom=0, top=1)
    plt.ylabel("Average actual accuracy")


print(f'# Plotting and saving data ... ', end=f'')
sys.stdout.flush()

# Create plots for every subjects
for s in subjects:
    calibration(s)
    plt.savefig(f'assets/plots/{s}_calibration.png')
    plt.close('all')


average_calibration()
plt.savefig(f'assets/plots/average_calibration.png')

print(f'Done!')
