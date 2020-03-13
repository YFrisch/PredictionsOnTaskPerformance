""" This file is used for evaluation and visualization of the brier score vs task score relationship.

All available evaluation data is read-in first.

Per default this script runs plot_rating_vs_brier(), averaging over all available subjects in the subject folder.
"""
import src.utils
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.data_reader import read_csv_files
from sklearn.linear_model import LinearRegression


__author__ = 'Yannik P. Frisch, Maximilian A. Gehrke'
__date__ = '12-03-2020'


# Set working directory to the top level of our project
src.utils.set_working_directory()


print("\n-------------------- EVALUATION --------------------")


# --------------- READ DATA --------------- #
subjects, subject_task_scores, subject_brier_scores, subject_probs, max_score = read_csv_files()


# --------------- EVALUATE DATA --------------- #
def plot_rating_vs_brier():
    """Evaluates the (Task Score) vs (Brier Score) relationship.

    Creates a scatter plot per task by it's mean task score (x-axis) and it's mean brier score (y-axis),
    averaged over all subjects.
    Then fits and plots a linear regression function on this relationship.

    :return: None
    """

    # -------------------- Gather Data --------------------

    # (task x subject) matrix of brier scores
    brier_scores_matrix = np.empty(shape=(1, len(subject_task_scores.get(subjects[0]))))
    # (task x subject) matrix of task scores
    task_scores_matrix = np.empty(shape=(1, len(subject_task_scores.get(subjects[0]))))
    for s in subjects:
        brier_scores_matrix = np.concatenate((brier_scores_matrix,
                                              np.array(list(subject_brier_scores.get(s).values())).reshape(1, -1)),
                                             axis=0)
        task_scores_matrix = np.concatenate((task_scores_matrix,
                                             np.array(list(subject_task_scores.get(s).values())).reshape(1, -1)),
                                            axis=0)
    # Delete initial rows
    brier_scores_matrix = np.copy(brier_scores_matrix[1:, :])
    task_scores_matrix = np.copy(task_scores_matrix[1:, :])

    # Get averages
    avg_brier_per_task = np.mean(brier_scores_matrix, axis=0).reshape(-1, 1)
    avg_score_per_task = np.mean(task_scores_matrix, axis=0).reshape(-1, 1)

    # Fit linear regression function
    reg = LinearRegression().fit(avg_score_per_task, avg_brier_per_task)
    reg.score(avg_score_per_task, avg_brier_per_task)

    # Plot values and linear fit
    plt.figure(figsize=(10, 5))
    plt.title("Task Score vs Brier Score")
    color = plt.get_cmap('rainbow', len(avg_brier_per_task))
    colors = color(np.linspace(0, 1, len(avg_brier_per_task)))
    for t in range(0, len(avg_brier_per_task)):
        task_color = colors[t].reshape(1, -1).squeeze()
        plt.scatter(avg_score_per_task[t], avg_brier_per_task[t], label=f'Task {t+1}', c=task_color)
    xs = np.linspace(min(avg_score_per_task), max(avg_score_per_task), 100)
    ys = np.array([reg.predict(x.reshape(1, -1)) for x in xs]).reshape(-1, 1)
    plt.plot(xs, ys, color='black', ls='--', label='Regression fit')
    plt.legend()
    plt.xlabel("Average task score")
    plt.ylabel("Average brier score")


print(f'# Plotting and saving data ... ', end=f'')
sys.stdout.flush()


plot_rating_vs_brier()
plt.savefig(f'assets/plots/brier_vs_rating.png')


print(f'Done!')
