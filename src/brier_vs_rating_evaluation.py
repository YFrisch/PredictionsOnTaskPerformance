import src.utils
from src.data_reader import read_csv_files

__author__ = 'Yannik Frisch'
__date__ = '08-03-2020'

# Set working directory to the top level of our project
src.utils.set_working_directory()


print("\n-------------------- EVALUATION --------------------")


# --------------- READ DATA --------------- #
subjects, subject_task_scores, subject_brier_scores, subject_probs, max_score = read_csv_files()


# --------------- EVALUATE DATA --------------- #
def plot_rating_vs_brier():
    ...
