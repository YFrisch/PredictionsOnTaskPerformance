"""
    Main Python File
"""

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '01-03-2020'

import os
import sys
import numpy as np
import pdf2image
import pandas as pd
from src.discrete_distribution_reader import DiscreteDistributionReader as DDR
from src.read_pdfs import extract_pdfs


'''
    Because the working directory is dependent on the configuration of the 
    python IDE each person is using, we make sure that the current
    working directory is set to the 'PredictionsOnTaskPerformance' folder
    and not the 'src' folder.
'''
current_folder = os.path.basename(os.getcwd())
if current_folder == f'src':
    os.chdir(f'..')


# --------------- Scoring Function --------------- #

def score(answers, points_per_task):
    """ This function returns a discrete score for an input numpy array of actual answer positions for a sorting task.
        E.g. if a subjects answer for a task is a sorting of [A, C, E, B, D], we pass the correct positions
        [1, 3, 5, 2, 4] to the function.
        The method then calculates the 2-norm between the right answer [1, 2, 3, 4, 5]
        and the given answer [1, 3, 5, 2, 4].
        The interval [0, max-2-norm] is split into an array of a fixed amount of equidistant numbers (e.g. 5 or 10).
        The discrete rating for this task is the 'id' of the closes value of that array compared to the norm
        of the answer.
        E.g. if the subjects answer got a norm of 3.16 and we have a max-2-norm of 6.32 (worst possible answer) and we
        split the task into 5 possible ratings (1-5 points), we get the intervals [0, 1.27, 2.53, 3.79, 5.06, 6.32].
        The function would return a rating of 5 - 3 = 2.
    """
    def find_nearest(array, value):
        """ Returns entry of array that is closest to value.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    task_norm = np.linalg.norm(x=(np.array([1, 2, 3, 4, 5]) - answers), ord=2)
    max_norm = np.linalg.norm(x=(np.array([1, 2, 3, 4, 5]) - np.array([5, 4, 3, 2, 1])), ord=2)
    # We use a min 0 points and a max of points_per_task
    intervals = np.linspace(0, max_norm, points_per_task+1)
    rating = points_per_task - np.where(intervals == find_nearest(intervals, task_norm))[0][0]
    return rating


# score(answers=np.array([1, 3, 5, 2, 4]), points_per_task=5)
# TODO: Use score function to calculate the brier score for the answers.

# --------------- Folder & File Names --------------- #

# Specify all the file suffixes (difference between folder name and file name)
# for each file that should be read. The pdfs will be read and annotated
# in the order the files appear in the array.
file_suffixes = [f'_p1.jpg', f'_p2.jpg', f'_p3.jpg']

# Getting all the subdirectory names in the subjects folder
subjects_folder_path = f'assets/subjects/'
subject_dirs_ = os.listdir(subjects_folder_path)

# Delete folders that do not start with "subject" (e.g. Apple hidden .DS_Store)
subjects = []
for subject_dir in subject_dirs_:
    if subject_dir.startswith("subject_"):
        subjects.append(subject_dir[8:])

print(subjects)

# --------------- Read PDFs --------------- #

# Iterate over all subjects and read the pdfs
for subject in subjects:

    # Create array with a seperate string for each file that we
    # want to read for this subject.
    images_array = []
    for fs in file_suffixes:
        images_array.append(f'{subjects_folder_path}'
                            f'subject_{subject}/raw/subject_{subject}{fs}')

    # First option: one pdf of the whole experiment
    # 1. page: explanation (incl. subject code), 2. - 4. page: experiment
    multiple_pdf_path = f'{subjects_folder_path}subject_{subject}' \
                        f'/raw/subject_{subject}.pdf'
    if os.path.exists(multiple_pdf_path):
        files = pdf2image.convert_from_path(multiple_pdf_path)
        # skip first page (only explanation)
        for i in range(1, len(files)):
            file = files[i]
            fs = file_suffixes[i-1]
            file.save(f'{multiple_pdf_path[:-4]}{fs}', f'jpeg')
    else:
        # Convert pdf to jpg if it is not already available in jpg
        # Note: The "poppler" package needs to be installed
        for img_path in images_array:
            if not os.path.exists(img_path):
                files = pdf2image.convert_from_path(f'{img_path[:-3]}pdf')
                files[0].save(img_path, f'jpeg')

    # Path where we save the probability density functions
    dst_path = f'{subjects_folder_path}subject_{subject}/pdfs/'

    # Create 'pdfs'-folder if it does not exist
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # Extract pdfs and save
    extract_pdfs(image_path_array=images_array, dst_folder=dst_path)


# --------------- Read Answer CSV --------------- #
'''
    Read the answers of the csv file and put them into a dictionary.
    The keys are 'task_1', 'task_2', ... and each entry is a list of
    the correct position of each answer for the specific task in
    ascending order.
    Example: subject wrote [A, C, E, B, D] correct would be 
    [A, B, C, D, E] so the list for that task is [1, 3, 5, 2, 4].
'''

# 2D array with subject dir as first entry and answers dictionary as second
subject_answers = []

for subject in subjects:
    path_to_csv = f'{subjects_folder_path}subject_{subject}/' \
                  f'subject_{subject}_answers.csv'
    if os.path.exists(path_to_csv):
        answers = pd.read_csv(path_to_csv, sep=',')
        answers = answers.set_index('task').T.to_dict(f'list')

        subject_answers.append([subject, answers])

# --------------- Save Scoring --------------- #

pts_per_task = 5

for subject, answers in subject_answers:
    scores = []
    analysis_path = f'{subjects_folder_path}subject_{subject}/analysis/'
    if not os.path.exists(analysis_path):
        os.mkdir(analysis_path)
    for i in range(1, 9):
        scores.append(score(answers[f'task_{i}'], pts_per_task))

    scores = pd.DataFrame(scores,
                          columns=[f'points (max {pts_per_task})'],
                          index=[f'task_1', f'task_2', f'task_3', f'task_4',
                                 f'task_5', f'task_6', f'task_7', f'task_8'])
    scores.to_csv(f'{analysis_path}task_scores.csv', sep=f',')

# --------------- Simulate Distribution --------------- #
#
dr = DDR(vpn_code='DTEA')
dr.plot(task_ids=[1, 2, 3, 4, 5, 6, 7, 8])




