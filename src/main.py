"""
    Main Python File
"""
from __future__ import print_function

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '01-03-2020'

import os
import sys
import numpy as np
import pdf2image
import pandas as pd
from src.discrete_distribution_reader import DiscreteDistributionReader as DDR
import src.utils
from src.read_answers import read_answers_from_csv
from src.scoring import apply_scoring
from src.pdf_reader import extract_pdfs

# --------------- SET PATH --------------- #

src.utils.set_working_directory()

# --------------- VARIABLES --------------- #

task_ids = [1, 2, 3, 4, 5, 6, 7, 8]
num_of_tasks = len(task_ids)

# Specify all the file suffixes (difference between folder name and file name)
# for each file that should be read. The pdfs will be read and annotated
# in the order the files appear in the array.
file_suffixes = [f'_p1.jpg', f'_p2.jpg', f'_p3.jpg']

subjects_folder_path = f'assets/subjects/'

pts_per_task = 5

# --------------- INITIALIZE --------------- #
print(f'Initializing ... ', end=f'')
sys.stdout.flush()

# Get subject codes
subjects = src.utils.extract_subject_codes_from_folders(subjects_folder_path)

# Create folder for saving the extracted probability density functions
src.utils.create_pdf_directories(subjects, subjects_folder_path)

# Create folder for saving the analysis results
src.utils.create_analysis_directories(subjects, subjects_folder_path)
print(f'Done!')

# --------------- PREPROCESSING --------------- #
print(f'Preprocessing ... ', end=f'')
sys.stdout.flush()
src.utils.convert_pdf_to_jpg(subjects, subjects_folder_path, file_suffixes)
extract_pdfs(subjects, subjects_folder_path, file_suffixes)
subject_answers = read_answers_from_csv(subjects, subjects_folder_path)
subject_task_scores = apply_scoring(subject_answers, task_ids,
                                    pts_per_task, subjects_folder_path)

print(f'Done!')
# --------------- Simulate Distribution --------------- #
for s in subjects:
    dr = DDR(vpn_code=s, task_scores=subject_task_scores.get(s))
    dr.plot(task_ids=[9])



