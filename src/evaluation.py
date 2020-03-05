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
        answers = pd.read_csv(path_to_csv, sep=',')
        #answers = answers.to_dict(orient=f'list')
        answers = answers.set_index('Unnamed: 0').T.to_dict(f'list')
        subject_brier_scores[subject] = answers

print("# Read in brier scores.")


# --------------- EVALUATE DATA --------------- #

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
plt.show()

