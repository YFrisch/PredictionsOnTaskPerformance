""" This file is used for several evaluations and plots of the calculated brier scores,
achieved task scores and assigned probabilities of the subjects.

All available evaluation data is read-in first.

Several methods to plot the data can be used.

Per default by calling this script, plot_vpn() is evaluated on all available subjects in the subject folder,
and bar plots of the averages for the task and brier scores are created and saved.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import src.utils
from src.data_reader import read_csv_files


__author__ = 'Yannik P. Frisch'
__date__ = '12-03-2020'


# Set working directory to the top level of our project
src.utils.set_working_directory()


print("\n-------------------- EVALUATION --------------------")


# --------------- READ DATA --------------- #
subjects, subject_task_scores, subject_brier_scores, subject_probs, max_score = read_csv_files()


# --------------- EVALUATE DATA --------------- #
def plot_average_task_scores():
    """This function calculates and plots average task scores for our experiment.

    This method creates and saves a figure with a bar plot of the achieved
    discrete rating per task, averaged over all subjects and a bar plot of the
    achieved discrete rating per subject, averaged over all tasks.
    The mean and standard deviation over the subject AND task axes is printed.

    :return: None
    """
    points_over_task = np.empty((1, len(subject_task_scores.get(subjects[0]))))
    for s in subjects:
        ts = np.array(list(subject_task_scores.get(s).values())).T
        points_over_task = np.concatenate((points_over_task, ts), axis=0)
    points_over_task = np.copy(points_over_task[1:, :])

    # Create a figure with two plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the mean of task scores grouped by tasks
    for i in range(0, points_over_task.shape[1]):
        axs[0].bar(x=i+1, height=np.mean(points_over_task[:, i]),
                   yerr=np.std(points_over_task[:, i]), color='red',
                   ecolor='black', align='center', alpha=0.3, capsize=5)

    # Plot the mean of task scores grouped by subjects
    for i in range(0, points_over_task.shape[0]):
        axs[1].bar(x=i+1, height=np.mean(points_over_task[i, :]),
                   yerr=np.std(points_over_task[i, :]), color='red',
                   ecolor='black', align='center', alpha=0.3, capsize=5)

    # Plot horizontal lines for the mean
    axs[0].hlines(np.mean(points_over_task), 0.6,
                  points_over_task.shape[1] + 0.4, color='orange')
    axs[1].hlines(np.mean(points_over_task), 0.6,
                  points_over_task.shape[0] + 0.4, color='orange')

    # Print average and standard deviation to the console
    print(f"The mean achieved task score over all tasks and subjects is "
          f"{np.round(np.mean(points_over_task), 4)} with a "
          f"standard deviation of {np.round(np.std(points_over_task), 4)}.")

    plt.suptitle("Average task scores")
    axs[0].set_title("Average points per task")
    axs[1].set_title("Average points per subject")
    axs[0].set_xlabel("Task ID")
    axs[1].set_xlabel("Subject ID")
    axs[0].set_ylabel("Average Points")
    axs[1].set_ylabel("Average Points")
    axs[1].set_xticks(np.arange(1, len(subjects)+1))
    plt.savefig(f'assets/plots/average_task_scores.png')
    plt.close('all')
    return None


def plot_average_brier_scores():
    """This method calculates and plots the average brier score for the subjects in our experiment.

    This method creates and saves a bar plot of the calculated brier score per
    task, averaged over all subjects and a bar plot of the mean brier score per
    subject, averaged over all tasks. The mean and standard deviation over the
    subject AND task axes is printed out.

    :return: None
    """
    brier_over_task = np.empty((1, len(subject_brier_scores.get(subjects[0]))))
    for s in subjects:
        bs = np.array(list(subject_brier_scores.get(s).values())).T
        brier_over_task = np.concatenate((brier_over_task, bs), axis=0)
    brier_over_task = np.copy(brier_over_task[1:, :])

    # Create a figure with two plots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the mean of task scores grouped by tasks
    for i in range(0, brier_over_task.shape[1]):
        axs[0].bar(x=i+1, height=np.mean(brier_over_task[:, i]),
                   yerr=np.std(brier_over_task[:, i]), color='green',
                   ecolor='black', align='center', alpha=0.3, capsize=5)

    # Plot the mean of task scores grouped by subjects
    for i in range(0, brier_over_task.shape[0]):
        axs[1].bar(x=i+1, height=np.mean(brier_over_task[i, :]),
                   yerr=np.std(brier_over_task[i, :]), color='green',
                   ecolor='black', align='center', alpha=0.3, capsize=5)

    # Plot horizontal lines for the mean
    axs[0].hlines(np.mean(brier_over_task), 0.6,
                  brier_over_task.shape[1] + 0.4, color='orange')
    axs[1].hlines(np.mean(brier_over_task), 0.6,
                  brier_over_task.shape[0] + 0.4, color='orange')

    # Print average and standard deviation to the console
    print(f"The mean brier score over all tasks and subjects is "
          f"{np.round(np.mean(brier_over_task), 4)} with "
          f"a standard deviation of {np.round(np.std(brier_over_task), 4)}.")

    plt.suptitle("Average brier scores")
    axs[0].set_title("Average brier score per task")
    axs[1].set_title("Average brier score per subject")
    axs[0].set_xlabel("Task ID")
    axs[1].set_xlabel("Subject ID")
    axs[0].set_ylabel("Avg. brier score")
    axs[1].set_ylabel("Avg. brier score")
    axs[1].set_xticks(np.arange(1, len(subjects) + 1))
    plt.savefig(f'assets/plots/average_brier_scores.png')
    plt.close('all')
    return None


def plot_subject(subject_code):
    """This method creates bar-plots for the task-score and brier-score per
    task for a given subject and an pyplot.imshow matrix plot for the assigned
    probabilities per normalized discrete rating per task (confidence).

    The plots are saved in a single figure with subplots.

    :param: subject_code: The 4-letter vpn-code of the subject
    :return: None
    """
    bs = np.array(list(subject_brier_scores.get(subject_code).values())).T.squeeze()
    ts = np.array(list(subject_task_scores.get(subject_code).values())).T.squeeze()
    prob_matrix = np.empty((1, 6))
    prob_dict = subject_probs.get(subject_code)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(0, 7):
        axs[0].bar(x=i+1, height=ts[i], color='red', align='center', alpha=0.3)
        axs[1].bar(x=i+1, height=bs[i], color='green', align='center', alpha=0.3)
        task_prob = np.array(prob_dict.get(i)).reshape((1, -1))
        prob_matrix = np.concatenate((prob_matrix, task_prob), axis=0)

    # Plot horizontal lines for the mean
    axs[0].hlines(np.mean(ts), 0.6,
                  ts.shape[0] + 0.4, color='orange')
    axs[1].hlines(np.mean(bs), 0.6,
                  bs.shape[0] + 0.4, color='orange')

    prob_matrix = np.copy(prob_matrix.T[:, 1:])
    # TODO: Scale imshow plot to same size as other plots and colorbar
    ims = axs[2].imshow(prob_matrix, cmap='Greys')
    axs[2].scatter(np.arange(0, 7), ts, marker='x', c='orange', alpha=1)
    axs[0].set_title(f"Points of {subject_code} per task")
    axs[0].set_xlabel("Task")
    axs[0].set_ylabel("Achieved Points")
    axs[0].set_ylim(top=5.0)
    axs[0].set_yticks(np.arange(0, 6))
    axs[1].set_title(f"Brier Scores of {subject_code}")
    axs[1].set_xlabel("Task")
    axs[1].set_ylabel("Brier Score")
    axs[1].set_ylim(top=2.0)
    axs[2].set_title(f"Estimated scores of {subject_code} per task")
    axs[2].set_xlabel("Task")
    axs[2].set_ylabel("Scores")
    plt.colorbar(ims, ax=axs[2])
    plt.suptitle(f"Subject {subject_code}")
    fig.subplots_adjust()


# Create plots with all subjects
plot_average_task_scores()
plot_average_brier_scores()

print(f'# Creating and saving plots ... ', end='')

# Create plots for every subjects
sys.stdout.flush()
for subject in subjects:
    plot_subject(subject)
    plt.savefig(f'assets/plots/{subject}_results.png')
    plt.close()

print(f'Done!')
