"""
Function for reading the answer CSV file.
Project: Predictions on Task Performance
"""

import os
import pandas as pd

__author__ = 'Maximilian A. Gehrke'
__date__ = '08-03-2020'


def read_answers_from_csv(subjects, subjects_folder_path):
    """
    Extracting the answers of a CSV file into a dictionary.

    Example: subject wrote [A, C, E, B, D], correct would be
    [A, B, C, D, E] so the list for that task is [1, 3, 5, 2, 4].
    This list is already available in the CSV file and we simply extract it.

    :param subjects: list of String; the subject codes
    :param subjects_folder_path: String; path to the subject folders
    :return: dictionary of dictionaries; first key is the subject code, second
        key is the task id in form of 'task_id' and the value is an  array of
        ints representing the answers of the subject for a specific task.
    """
    # Dictionary of dictionaries
    subject_answers = {}

    for subject in subjects:
        path_to_csv = f'{subjects_folder_path}subject_{subject}' \
                      f'/raw/subject_{subject}_answers.csv'
        if os.path.exists(path_to_csv):
            answers = pd.read_csv(path_to_csv, sep=',')
            answers = answers.set_index('task').T.to_dict(f'list')

            subject_answers[subject] = answers

    return subject_answers
