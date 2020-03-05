import os
import pdf2image


def set_working_directory():
    """
        TODO: Check
        The working directory is dependent on the configuration of the 
        python IDE. Here we make sure that the current working directory is set to
        the 'PredictionsOnTaskPerformance' folder and not the 'src' folder.
    """
    current_folder = os.path.basename(os.getcwd())
    if current_folder == f'src':
        os.chdir(f'..')


def create_analysis_directories(subjects, subjects_folder_path):
    """
    TODO
    :param subjects:
    :param subjects_folder_path:
    :return:
    """
    for subject in subjects:
        analysis_path = f'{subjects_folder_path}subject_{subject}/analysis/'
        if not os.path.exists(analysis_path):
            os.mkdir(analysis_path)
    return True


def create_pdf_directories(subjects, subjects_folder_path):
    """
    TODO Path where we save the probability density functions
    :param subjects:
    :param subjects_folder_path:
    :return:
    """
    for subject in subjects:
        pdf_path = f'{subjects_folder_path}subject_{subject}/pdfs/'
        if not os.path.exists(pdf_path):
            os.mkdir(pdf_path)
    return True


def extract_subject_codes_from_folders(subjects_folder_path):
    """
    TODO
    :param subjects_folder_path:
    :return:
    """
    # Getting all the subdirectory names in the subjects folder
    subject_dirs_ = os.listdir(subjects_folder_path)

    # Delete folders that do not start with "subject"
    # (e.g. Apples hidden .DS_Store)
    subjects = []
    for subject_dir in subject_dirs_:
        if subject_dir.startswith("subject_"):
            subjects.append(subject_dir[8:])

    return subjects


def convert_pdf_to_jpg(subjects, subjects_folder_path, file_suffixes):
    """

    :param subjects:
    :param subjects_folder_path:
    :param file_suffixes:
    :return:
    """
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
