""" This file contains diverse small utility functions. """

import os
import numpy as np

import pdf2image

__author__ = 'Maximilian A. Gehrke'
__date__ = '08-03-2020'


def set_working_directory():
    """
    Setting the working directory to the 'PredictionsOnTaskPerformance'
    folder, if it is currently in the 'src' folder. This step is
    necessary as the working directory is dependent on the configuration
    of the Python IDE.
    """
    current_folder = os.path.basename(os.getcwd())
    if current_folder == f'src':
        os.chdir(f'..')


def extract_subject_codes_from_folders(subjects_folder_path):
    """
    Extract the subject codes by taking all the folder names which start with
    'subject_' at the given location and extracting the subject codes from
    them. Our naming convention for subjects is 'subject_' + 'SUBJECTCODE'.

    :param subjects_folder_path: String; path to the subject folders
    :return: String array containing the subject codes.
    """
    # Getting all subdirectories of the subjects folder
    subject_dirs_ = os.listdir(subjects_folder_path)

    # Delete folders that do not start with 'subject_'
    # (e.g. Apples hidden .DS_Store)
    # and extract the subject code from the folder name.
    subjects = []
    for subject_dir in subject_dirs_:
        if subject_dir.startswith("subject_"):
            subjects.append(subject_dir[8:])

    return subjects


def create_folder(folder_path):
    """
    Create a folder at the given path if it does not already exist.
    :param folder_path: The path of the desired folder
    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def create_analysis_directories(subjects, subjects_folder_path):
    """
    Create a folder named analysis for each subject.
        (This function is hard coded).
    :param subjects: String array; all subject codes
    :param subjects_folder_path: String; the path to the subject's folders.
    """
    for subject in subjects:
        analysis_path = f'{subjects_folder_path}subject_{subject}/analysis/'
        create_folder(analysis_path)


def create_pdf_directories(subjects, subjects_folder_path):
    """
    Create a folder named pdf for each subject (hard coded), where we will
    save all the extracted probability density functions.
    :param subjects: String array; all subject codes
    :param subjects_folder_path: String; the path to the subject's folders.
    """
    for subject in subjects:
        pdf_path = f'{subjects_folder_path}subject_{subject}/pdfs/'
        create_folder(pdf_path)


def convert_pdf_to_jpg(subjects, subjects_folder_path, file_suffixes):
    """
    Convert PDF pages to JPEG images. The paths are hard coded for this
    experiment. This function iterates over all subject folders and checks
    whether JPEG's of the questionnaire already exist in the 'raw' folder.
    If so, the function does nothing.

    Otherwise the function checks if a PDF is available in the 'raw' folder
    with the name 'subject_SUBJECTCODE.pdf'. If so, this PDF is read, the
    first page is discarded and the remaining pages are converted into JPEG's.

    If there is no PDF called 'subject_SUBJECTCODE.pdf' in 'raw', the function
    expects separate PDF files with one page only, that follow our naming
    convention.

    NOTE: This function uses the pdf2image library which might need the
    'poppler' package to be installed.


    :param subjects: String array of subject codes.
    :param subjects_folder_path: String; path to the subjects folder
    :param file_suffixes: String array with the file endings of the
        PDF / JPEG's we want to analyse.
    """
    for subject in subjects:
        # Create an array with a separate string for each
        # file that we want to read for this subject.
        images_array = []
        for fs in file_suffixes:
            images_array.append(f'{subjects_folder_path}'
                                f'subject_{subject}/raw/subject_{subject}{fs}')

        # One PDF for the complete questionnaire
        # First page: explanation (incl. subject code)
        # Second to fourth page: experiment
        multiple_pdf_path = f'{subjects_folder_path}subject_{subject}' \
                            f'/raw/subject_{subject}.pdf'
        if os.path.exists(multiple_pdf_path):
            files = pdf2image.convert_from_path(multiple_pdf_path)
            # Skip first page
            for i in range(1, len(files)):
                file = files[i]
                fs = file_suffixes[i-1]
                file.save(f'{multiple_pdf_path[:-4]}{fs}', f'jpeg')

        # Convert single page PDFs into JPEGs
        else:
            for img_path in images_array:
                if not os.path.exists(img_path):
                    files = pdf2image.convert_from_path(f'{img_path[:-3]}pdf')
                    files[0].save(img_path, f'jpeg')


def find_nearest(array, value):
    """
    Find the entry of the array that is closest to a given value.
    :param array: list or array of values
    :param value: int; a number
    :return: int; the number in the array that is closest to the given value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
