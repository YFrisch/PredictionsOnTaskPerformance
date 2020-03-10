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
    """ The calibration() function creates and saves a calibration curve for the given subject.

    The average probability per task is compared with the overall confidence assigned by the subject.
    #TODO: Average across subjects

    :param subject_code:
    :return:
    """
    # Read in data
    ts = np.array(list(subject_task_scores.get(subject_code).values())).T.squeeze()
    prob_matrix = np.empty((1, 6))
    prob_dict = subject_probs.get(subject_code)
    for i in range(0, 9):
        task_prob = np.array(prob_dict.get(i)).reshape((1, -1))
        prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
    overall_probs = np.copy(prob_matrix.T[:, -1])

    received_points_percentage = np.zeros(shape=(6, 1))
    for i in range(0, len(ts)):
        received_points_percentage[ts[i]] += 1
    received_points_percentage = (received_points_percentage/len(ts)).T.squeeze()

    plt.figure(figsize=(15, 5))
    plt.title(f'Overall confidence of {subject_code}')
    for i in range(0, len(overall_probs)):
        op_i = overall_probs[i]
        rpp_i = received_points_percentage[i]
        plt.scatter(i, op_i, color='orange', s=50.0, alpha=1.0, label='Confidence')
        plt.scatter(i, rpp_i, color='black', s=50.0, alpha=1.0, label='Actual')
        if rpp_i - op_i < 0:
            plt.plot([i, i], [rpp_i, op_i], 'k-', color='red', label='Overconfidence')
        elif rpp_i - op_i > 0:
            plt.plot([i, i], [rpp_i, op_i], 'k-', color='green', label='Underconfidence')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.plot(range(0, len(overall_probs)), overall_probs, color='orange')
    plt.plot(range(0, len(received_points_percentage)), received_points_percentage, color='black')
    plt.fill_between(range(0, len(received_points_percentage)), overall_probs, received_points_percentage,
                     where=overall_probs >= received_points_percentage, facecolor='red', alpha=0.7, interpolate=True)
    plt.fill_between(range(0, len(received_points_percentage)), overall_probs, received_points_percentage,
                     where=overall_probs <= received_points_percentage, facecolor='green', alpha=0.7, interpolate=True)
    plt.xlabel('Discrete Points')
    plt.ylabel('Probability')


def mod_calibration(subject_codes):
    """ The calibration() function creates and saves a calibration curve for the given subject.

    The average probability per task is compared with the overall confidence assigned by the subject.
    #TODO: Average across subjects

    :param subject_code:
    :return:
    """
    # Read in data
    ts = defaultdict(np.ndarray)
    ops = defaultdict(np.ndarray)
    for s in range(0, len(subject_codes)):
        ts[s] = np.array(list(subject_task_scores.get(s).values())).T.squeeze()
        prob_matrix = np.empty((1, 6))
        prob_dict = subject_probs.get(s)
        for i in range(0, 9):
            task_prob = np.array(prob_dict.get(i)).reshape((1, -1))
            prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
        overall_probs = np.copy(prob_matrix.T[:, -1])
        ops[s] = overall_probs

    received_points_percentage = np.zeros(shape=(6, len(subject_codes)))
    for s in range(0, len(subject_codes)):
        for i in range(0, len(ts)):
            received_points_percentage[ts[i], s] += 1
        received_points_percentage[:, s] = (received_points_percentage[:, s]/len(ts[s])).T.squeeze()
    averaged_rpp = np.mean(received_points_percentage, axis=1)
    averaged_op = np.mean(ops, axis=1)
    print(f"a_rpp: {averaged_rpp.shape} a_op: {averaged_op.shape}")

    plt.figure(figsize=(15, 5))
    plt.title(f'Overall average calibration')
    for i in range(0, len(averaged_op)):
        op_i = averaged_op[i]
        rpp_i = averaged_rpp[i]
        plt.scatter(i, op_i, color='orange', s=50.0, alpha=1.0, label='Confidence')
        plt.scatter(i, rpp_i, color='black', s=50.0, alpha=1.0, label='Actual')
        if rpp_i - op_i < 0:
            plt.plot([i, i], [rpp_i, op_i], 'k-', color='red', label='Overconfidence')
        elif rpp_i - op_i > 0:
            plt.plot([i, i], [rpp_i, op_i], 'k-', color='green', label='Underconfidence')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.plot(range(0, len(averaged_op)), averaged_op, color='orange')
    plt.plot(range(0, len(averaged_rpp)), received_points_percentage, color='black')
    plt.fill_between(range(0, len(received_points_percentage)), averaged_op, averaged_rpp,
                     where=averaged_op >= averaged_rpp, facecolor='red', alpha=0.7, interpolate=True)
    plt.fill_between(range(0, len(averaged_rpp)), averaged_op, averaged_rpp,
                     where=averaged_op <= averaged_rpp, facecolor='green', alpha=0.7, interpolate=True)
    plt.xlabel('Discrete Points')
    plt.ylabel('Probability')


# Create plots for every subjects
for subject in subjects:
    calibration(subject)
    plt.savefig(f'assets/plots/{subject}_calibration.png')
    plt.close()


print(f'# Save Plots ... Done!')