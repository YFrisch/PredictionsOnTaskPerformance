"""
Executing this file processes the data gathered by our questionnaire.

For our project 'Predicitons on Task Performance', we designed a
questionnaire where subjects execute little tasks and draw probability
density functions over their confidence that they get a specific amount
of points. After collecting answers from different subjects, the filled
questionnaires are scanned and the answers are written into CSV files.

This file loads the questionnaires, converts PDFs into JPEGs, extracts
the probability density functions and imports the subjects answers. After-
wards we apply a scoring metric to the subjects answers and read off the
probabilities from the probability density functions. With these two features
we calculate the Brier score, which measures the accuracy of the subjects
predictions.

All important information is saved to the disk and is available for inspection.
In the file 'evaluation.py' we use this data to calculate statistical measures
(e.g. mean and standard deviation) and create plots.
"""

import sys

import src.utils
from src.scoring import apply_scoring
from src.pdf_reader import extract_pdfs
from src.read_answers import read_answers_from_csv
from src.discrete_distribution_reader import DiscreteDistributionReader as DDR

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '01-03-2020'

# --------------- SET PATH --------------- #

src.utils.set_working_directory()

# --------------- VARIABLES --------------- #

# The 'id's of the tasks we want to analyse
task_ids = [1, 2, 3, 4, 5, 6, 7, 8]
num_of_tasks = len(task_ids)

# We will create a separate file for each page of the experiment
# following the structure 'subject_SUBJECTCODE_suffix'.
file_suffixes = [f'_p1.jpg', f'_p2.jpg', f'_p3.jpg']

# Path where the subject folders are located
subjects_folder_path = f'assets/subjects/'

# The maximum amount of points that can be achieved for each task
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

# Convert PDF files into JPEGs if not already available in JPEG
src.utils.convert_pdf_to_jpg(subjects, subjects_folder_path, file_suffixes)

# Get the subject answers from CSV and save as a dictionary of dictionaries
subject_answers = read_answers_from_csv(subjects, subjects_folder_path)

print(f'Done!')

# --------------- READ pdf's --------------- #
print(f'Extracting density functions ... ', end=f'')
sys.stdout.flush()

# Extract the probability density functions and save them as images
extract_pdfs(subjects, subjects_folder_path, file_suffixes)

print(f'Done!')

# --------------- SCORING --------------- #
print(f'Scoring subject answers ... ', end=f'')
sys.stdout.flush()

# Apply our scoring function, save it to disk and create a dictionary
subject_task_scores = apply_scoring(subject_answers, task_ids,
                                    pts_per_task, subjects_folder_path)

print(f'Done!')

# --------------- Getting probabilites & Calc Brier score --------------- #
print(f'Reading distributions and calculating brier scores ... ', end=f'')
sys.stdout.flush()

for s in subjects:
    dr = DDR(vpn_code=s, task_scores=subject_task_scores.get(s))

print(f'Done!')


