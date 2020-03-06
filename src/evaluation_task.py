""" This file is used for several evaluations and plots of the achieved task scores of the subjects.
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
subject_task_scores = {}

for subject in subjects:
    path_to_csv = BASE_DIR + f'/assets/subjects/subject_{subject}/' \
                  f'analysis/{subject}_task_scores.csv'
    if os.path.exists(path_to_csv):
        scores = pd.read_csv(path_to_csv, sep=',')
        scores = scores.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_task_scores[subject] = scores

print("# Read in brier scores.")


# --------------- EVALUATE DATA --------------- #
def plot_average_points_per_task():
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
    plt.show()
    return None


def vpn_points_confidence(vpn_code):
    ts = np.array(list(subject_task_scores.get(vpn_code).values())).T.squeeze()
    plt.figure()
    for i in range(0, len(ts)):
        plt.bar(x=i+1, height=ts[i], color='red', align='center', alpha=0.3)
    plt.title(f"Points of {vpn_code} per task")
    plt.xlabel("Task ID")
    plt.yticks(np.arange(0, 6, 1))
    plt.ylabel("Points")
    plt.show()
    return None


vpn_points_confidence(f'UENN')

