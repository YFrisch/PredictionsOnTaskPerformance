""" This file is used for several evaluations and plots of the calculated brier scores of the subjects.
"""

# --------------- IMPORTS ETC --------------- #

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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

# Dictionary of dictionaries
subject_brier_scores = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_brier_scores.csv'
    if os.path.exists(path_to_csv):
        pandas_frame = pd.read_csv(path_to_csv, sep=',')
        bs = pandas_frame.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_brier_scores[subject] = bs

print("# Read in brier scores.")

# Dictionary of probabilities of task
subject_probs = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_probabilities.csv'
    if os.path.exists(path_to_csv):
        pandas_frame = pd.read_csv(path_to_csv, sep=',')
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
        scores = scores.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_task_scores[subject] = scores

print("# Read in achieved task scores.")


# --------------- EVALUATE DATA --------------- #
def plot_average_task_scores():
    points_over_task = np.empty((1, len(subject_task_scores.get(subjects[0]))))
    for s in subjects:
        ts = np.array(list(subject_task_scores.get(s).values())).T
        points_over_task = np.concatenate((points_over_task, ts), axis=0)

    plt.figure()
    for i in range(0, points_over_task.shape[1]):
        plt.bar(x=i+1, height=np.mean(points_over_task[:, i]), yerr=np.std(points_over_task[:, i]),
                color='blue', ecolor='black', align='center', alpha=0.3, capsize=10)
    plt.title("Average points per task.")
    plt.xlabel("Task ID")
    plt.ylabel("Average Points")
    return None


def plot_vpn_task_scores(vpn_code):
    ts = np.array(list(subject_task_scores.get(vpn_code).values())).T.squeeze()
    plt.figure()
    for i in range(0, len(ts)):
        plt.bar(x=i+1, height=ts[i], color='red', align='center', alpha=0.3)
    plt.title(f"Points of {vpn_code} per task")
    plt.xlabel("Task ID")
    plt.yticks(np.arange(0, 6, 1))
    plt.ylabel("Points")
    return None


def plot_average_brier_scores():
    brier_over_task = np.empty((1, len(subject_brier_scores.get(subjects[0]))))
    for s in subjects:
        bs = np.array(list(subject_brier_scores.get(s).values())).T
        brier_over_task = np.concatenate((brier_over_task, bs), axis=0)

    plt.figure()
    for i in range(0, brier_over_task.shape[1]):
        plt.bar(x=i+1, height=np.mean(brier_over_task[:, i]), yerr=np.std(brier_over_task[:, i]),
                color='blue', ecolor='black', align='center', alpha=0.3, capsize=10)
    plt.title("Average brier score per task.")
    plt.xlabel("Task ID")
    plt.ylabel("Avg. Brier Score")
    return None


def plot_vpn_brier_scores(vpn_code):
    bs = np.array(list(subject_brier_scores.get(vpn_code).values())).T.squeeze()
    plt.figure()
    for i in range(0, len(bs)):
        plt.bar(x=i+1, height=bs[i], color='green', align='center', alpha=0.3)
    plt.title(f"Brier Scores of {vpn_code}")
    plt.xlabel("Task ID")
    plt.ylabel("Brier Score")
    return None


def plot_vpn_probabilities(vpn_code):
    prob_matrix = np.empty((1, 6))
    prob_dict = subject_probs.get(vpn_code)
    for i in range(0, len(prob_dict)):
        task_prob = np.array(prob_dict.get(i)).reshape((1, -1))
        prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)
    prob_matrix = np.copy(prob_matrix.T[:, 1:])
    plt.imshow(prob_matrix)
    # TODO: Rename x-axis by range(1, 10) (r.n. it's range(0, 9)
    plt.title(f"Estimated scores of {vpn_code} per task")
    plt.xlabel("Task")
    plt.ylabel("Score")
    plt.colorbar()
    return None


# TODO: Get that shit in subplots of ONE figure
#plot_vpn_probabilities(f'ATDA')
#plot_vpn_task_scores(f'ATDA')
#plot_vpn_brier_scores(f'ATDA')
plot_average_task_scores()
plot_average_brier_scores()
plt.show()
