""" This file is used to read-in our results stored in .csv files.

It enables to read-in all data that is saved by discrete_distribution_reader.py and needed for our evaluation files.
Also reads-in the manually saved ..._task_scores.csv file containing all achieved points per task for a subject.
"""
import sys
import os
import pandas as pd
import src.utils


__author__ = 'Maximilian A. Gehrke, Yannik P. Frisch, '
__date__ = '12-03-2020'


def read_csv_files():
    """ Reads-in the .csv files and returns dictionaries containing the data.

    :return: subjects: Dictionary containing all subject code strings
    :return: subject_task_scores: Dictionary containing lists of the achieved ratings, per subject
    :return: subject_brier_scores: Dictionary containing lists of calculated brier scores, per subject
    :return: subject_probs: Dictionary of probability matrices for each subjects.
                Matrices have dimensions (|Tasks| x |Ratings|)
    """
    print(f'# Reading data ... ', end='')
    sys.stdout.flush()
    subjects_folder_path = f'assets/subjects/'
    subjects = src.utils.extract_subject_codes_from_folders(subjects_folder_path)

    # Create plots folder if it is not there
    src.utils.create_folder(f'assets/plots/')

    subject_brier_scores = {}
    subject_probs = {}
    subject_task_scores = {}
    max_score = 5

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
        pandas_frame = pandas_frame.drop([7, 8])  # Drop task 8 and overall pdf
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
    return subjects, subject_task_scores, subject_brier_scores, subject_probs, max_score


def read_answers_from_csv(subjects, subjects_folder_path):
    """Extracting the answers of a CSV file into a dictionary.

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
