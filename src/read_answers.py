import os
import pandas as pd


def read_answers_from_csv(subjects, subjects_folder_path):
    '''
        TODO
        Read the answers of the csv file and put them into a dictionary.
        The keys are 'task_1', 'task_2', ... and each entry is a list of
        the correct position of each answer for the specific task in
        ascending order.
        Example: subject wrote [A, C, E, B, D] correct would be
        [A, B, C, D, E] so the list for that task is [1, 3, 5, 2, 4].
    '''

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