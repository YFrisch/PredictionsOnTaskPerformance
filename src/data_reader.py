import sys
import pandas as pd
import src.utils


__author__ = 'Yannik Frisch'
__date__ = '08-03-2020'


def read_csv_files():
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
