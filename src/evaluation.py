""" This file is used for several evaluations and plots of the calculated brier scores,
    achieved task scores and assigned probabilities of the subjects.

    All available evaluation data is read-in first.

    Several methods to plot the data can be used.

    Per default by calling this script, plot_vpn() is evaluated on all available subjects in the subject folder,
    and bar plots of the averages for the task and brier scores are created and saved.
"""

__author__ = 'Yannik Frisch'
__date__ = '08-03-2020'

# --------------- IMPORTS ETC --------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import src.utils

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))


# --------------- READ-IN DATA --------------- #

print("\n-------------------- EVALUATION --------------------")
# Getting all the subdirectory names in the subjects folder
subjects_folder_path = BASE_DIR + f'/assets/subjects/'
subject_dirs_ = os.listdir(subjects_folder_path)

# Delete folders that do not start with "subject" (e.g. Apple hidden .DS_Store)
subjects = []
for subject_dir in subject_dirs_:
    if subject_dir.startswith("subject_"):
        subjects.append(subject_dir[8:])

print("# Read in subjects: ", subjects)

# TODO: The 3 following for-loops are very redundant and could get combined

# Dictionary of dictionaries
subject_brier_scores = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_brier_scores.csv'
    if os.path.exists(path_to_csv):
        pandas_frame = pd.read_csv(path_to_csv, sep=',')
        # pandas_frame = pandas_frame.drop(7)
        pandas_bs = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_brier_scores[subject] = pandas_bs

print("# Read in brier scores.")

# Dictionary of probabilities of task
subject_probs = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_probabilities.csv'
    if os.path.exists(path_to_csv):
        pandas_frame = pd.read_csv(path_to_csv, sep=',')
        # pandas_frame = pandas_frame.drop(8)
        probs = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_probs[subject] = probs

print("# Read in estimated probabilities.")

# Dictionary of dictionaries
subject_task_scores = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_task_scores.csv'
    if os.path.exists(path_to_csv):
        scores = pd.read_csv(path_to_csv, sep=',')
        # scores = scores.drop(7)
        scores = scores.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_task_scores[subject] = scores

print("# Read in achieved task scores.")


# --------------- EVALUATE DATA --------------- #
def plot_average_task_scores():
    """ This method creates and saves a bar plot of the achieved discrete rating per task,
        averaged over all subjects.
    :return: None
    """
    # TODO: Fix; The scores are summed up somewhere / somehow
    points_over_task = np.empty((1, len(subject_task_scores.get(subjects[0]))))
    for s in subjects:
        ts = np.array(list(subject_task_scores.get(s).values())).T
        points_over_task = np.concatenate((points_over_task, ts), axis=0)

    plt.figure()
    for i in range(0, points_over_task.shape[1]):
        plt.bar(x=i+1, height=np.mean(points_over_task[:, i]), yerr=np.std(points_over_task[:, i]),
                color='blue', ecolor='black', align='center', alpha=0.3, capsize=10)
    plt.hlines(np.mean(points_over_task), 0.6, points_over_task.shape[1]+0.4, color='orange')
    plt.title("Average points per task.")
    plt.xlabel("Task ID")
    plt.ylabel("Average Points")
    plt.savefig(BASE_DIR + f'/assets/results/average_task_scores.png')
    plt.close('all')
    return None


def plot_average_brier_scores():
    """ This method creates and saves a bar plot of the calculated brier score per task,
        averaged over all subjects.
    :return: None
    """
    brier_over_task = np.empty((1, len(subject_brier_scores.get(subjects[0]))))
    for s in subjects:
        bs = np.array(list(subject_brier_scores.get(s).values())).T
        brier_over_task = np.concatenate((brier_over_task, bs), axis=0)

    plt.figure()
    for i in range(0, brier_over_task.shape[1]):
        plt.bar(x=i+1, height=np.mean(brier_over_task[:, i]), yerr=np.std(brier_over_task[:, i]),
                color='blue', ecolor='black', align='center', alpha=0.3, capsize=10)
    plt.hlines(np.mean(brier_over_task), 0.6, brier_over_task.shape[1]+0.4, color='orange')
    plt.title("Average brier score per task.")
    plt.xlabel("Task ID")
    plt.ylabel("Avg. Brier Score")
    plt.savefig(BASE_DIR + f'/assets/results/average_brier_scores.png')
    plt.close('all')
    return None


def plot_vpn(vpn_code):
    """ This method creates bar-plots for the task-score and brier-score per task for a given subject and an
        pyplot.imshow matrix plot for the assigned probabilities per normalized discrete rating per task (confidence).
        The graphics are saved in a single figure with subplots.
     :param: vpn_code: The 4-letter vpn-code of the subject
     :return: None
     """
    bs = np.array(list(subject_brier_scores.get(vpn_code).values())).T.squeeze()
    ts = np.array(list(subject_task_scores.get(vpn_code).values())).T.squeeze()
    prob_matrix = np.empty((1, 6))
    prob_dict = subject_probs.get(vpn_code)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0, 8):
        axs[0].bar(x=i+1, height=ts[i], color='red', align='center', alpha=0.3)
        axs[1].bar(x=i+1, height=bs[i], color='green', align='center', alpha=0.3)
        task_prob = np.array(prob_dict.get(i)).reshape((1, -1))
        prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
    prob_matrix = np.copy(prob_matrix.T[:, 1:])
    # TODO: Scale imshow plot to same size as other plots and colorbar
    ims = axs[2].imshow(prob_matrix, cmap='Greys')
    axs[0].set_title(f"Points of {vpn_code} per task")
    axs[0].set_xlabel("Task")
    axs[0].set_ylabel("Achieved Points")
    axs[1].set_title(f"Brier Scores of {vpn_code}")
    axs[1].set_xlabel("Task")
    axs[1].set_ylabel("Brier Score")
    axs[2].set_title(f"Estimated scores of {vpn_code} per task")
    axs[2].set_xlabel("Task")
    axs[2].set_ylabel("Scores")
    plt.colorbar(ims, ax=axs[2])
    plt.suptitle(f"Subject {vpn_code}")
    fig.subplots_adjust()


src.utils.create_folder(BASE_DIR + f'/assets/results/')

for subject in subjects:
    plot_vpn(subject)
    plt.savefig(BASE_DIR + f'/assets/results/{subject}_results.png')
    plt.close()

plot_average_task_scores()
plot_average_brier_scores()

print("# Saved results.")
