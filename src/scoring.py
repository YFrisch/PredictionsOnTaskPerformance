"""
Defining and applying scoring functions.
Project: Predictions on Task Performance
"""

import numpy as np
import pandas as pd

from src.utils import find_nearest

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '08-03-2020'


def score_sorting_task(answers, points_per_task):
    """
    Scoring the subject answers of a sorting task.

    This function returns a discrete score for an input numpy array of actual
    answer positions for a sorting task. E.g. if a subjects answer for a task
    is a sorting of [A, C, E, B, D], we pass the correct positions
    [1, 3, 5, 2, 4] to the function.

    The method then calculates the L2-norm between the correct answer
    [1, 2, 3, 4, 5] and the given answer [1, 3, 5, 2, 4]. To allocate discrete
    score values, we calculate the worst possible L2-norm (= max-L2-norm) and
    split the interval [0, max-L2-norm] into an array of a fixed amount of
    equidistant numbers (e.g. 5 or 10). The discrete rating is then the 'id'
    of the closest value of that array compared to the L2-norm of the answer.

    Example: if the subject's answer got a norm of 3.46 and we have a
    max-L2-norm of 6.32 (worst possible answer) and we split the task into 5
    possible ratings (0-5 points), we get the interval [0, 1.27, 2.53, 3.79,
    5.06, 6.32]. Because the norm of the subject's answer is closest to the
    forth entry ('id' = 3), the function would return a score of 5 - 3 = 2.

    :param answers: array with the five answers to a specific task.
    :param points_per_task: int; the maximum number of points that can be
        achieved for a task.
    :return: int; the score for the task
    """
    correct_order = np.array([1, 2, 3, 4, 5])
    worst_order = np.array([5, 4, 3, 2, 1])

    task_norm = np.linalg.norm(x=(correct_order - answers), ord=2)
    max_norm = np.linalg.norm(x=(correct_order - worst_order), ord=2)

    print(max_norm)

    # Points can lie in the range of [0, points_per_task]
    intervals = np.linspace(0, max_norm, points_per_task+1)
    print(intervals)
    nearest_value = find_nearest(intervals, task_norm)
    rating = points_per_task - np.where(intervals == nearest_value)[0][0]
    return rating


def apply_scoring(subject_answers, task_ids, pts_per_task,
                  subjects_folder_path):
    """
    Apply scoring to the answers of the subjects.

    :param subject_answers: dictionary of dictionaries; first key is the
        subject code, second key is the task id in form of 'task_id' and
        the value is an array of ints representing the answers of the subject
        at the specific task.
    :param task_ids: list of int; the id's of the tasks we want to look at
    :param pts_per_task: int; the maximum number of points that can be
        achieved for a task.
    :param subjects_folder_path: String; path to the subject folders
    :return: dictionary; keys are subject codes and values are int arrays of
        the scoring for each task.
    """
    subject_task_scores = {}
    for subject, answers in subject_answers.items():

        # Extract scores and save them in a dictionary with
        # key = subject code and value = array of scores.
        scores = []
        for i in task_ids:
            score = score_sorting_task(answers[f'task_{i}'], pts_per_task)
            scores.append(score)
        subject_task_scores[subject] = scores

        # Save scores to file
        indices = [f'task_{i}' for i in task_ids]
        column_names = [f'points (max {pts_per_task})']
        scores_df = pd.DataFrame(scores, columns=column_names, index=indices)
        save_path = f'{subjects_folder_path}subject_{subject}' \
                    f'/analysis/{subject}_task_scores.csv'
        scores_df.to_csv(save_path, sep=f',')

    return subject_task_scores
