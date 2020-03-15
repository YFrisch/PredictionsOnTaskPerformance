"""Extracting probability density functions and saving them to disk.

This file extracts the probability density functions of our questionnaire
and saves them as images to the disk. We use computer vision to detect the
contour of our square coordinate systems after we detected vertical and
horizontal lines.

The function is tailored to the design of our questionnaire.
"""

import numpy as np
import cv2

__author__ = 'Yannik Frisch, Maximilian A. Gehrke'
__date__ = '01-03-2020'


def sort_contours(contours, method=f'left-to-right'):
    """Sorting contours extracted with OpenCV.

    Partly from: https://www.pyimagesearch.com/2015/04/20/
    sorting-contours-using-python-and-opencv/

    :param contours: array; contours that we extracted using findContours by
        OpenCV.
    :param method: String; describing the sorting order, one of the following
        ['left-to-right', 'right-to-left', 'top-to-bottom', 'bottom-to-top']
    :return: sorted contours, sorted bounding boxes
    """
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # i = 0 selects the x-coordinates of the bounding boxes (left <-> right)
    # i = 1 selects the y-coordinates of the bounding boxes (top <-> bottom)
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # Compute bounding boxes of the contours
    bb = [cv2.boundingRect(c) for c in contours]

    # Sort bounding boxes and contours in the way we want
    (contours, bb) = zip(*sorted(zip(contours, bb), key=lambda b: b[1][i],
                                 reverse=reverse))

    return contours, bb


def extract_pdfs_(image_path_array, dst_folder, debugging=False):
    """Extract probability density functions for given files.

    This function extracts all probability density functions that can be
    found in the given files and saves them in the 'dst_folder'. The pdf's
    are annotated in the order the files are received from top to bottom
    follwing the formula 'pdf_task_<counter>', where counter is a running
    counter starting from 1.

    Partly from: https://medium.com/coinmonks/
    a-box-detection-algorithm-for-any-image-containing-boxes-756c15d7ed26

    :param image_path_array: String array; contains all the files we want
        to extract density functions from (ordering matters!).
    :param dst_folder: string; path to the folder where we store the extracted
        density function images.
    :param debugging: bool; set to True if intermediate prints are desired
    """

    # This counter sets the file name of the current pdf
    pdf_task_counter = 0

    for im_num, im_path in enumerate(image_path_array):

        # Read the image in grayscale (0 == grayscale)
        img = cv2.imread(im_path, 0)

        # Get height and width
        im_height, im_width = img.shape[:2]

        # Thresholding the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Invert the image
        img_bin = 255 - img_bin

        if debugging:
            cv2.imwrite(f'{dst_folder}debug_im{im_num}_binary.jpg', img_bin)

        # ------------------------ KERNEL ----------------------------------- #

        # Defining a kernel length
        ker_len = np.array(img).shape[1] // 80

        # A verticle kernel of (1 x kernel_length)
        # which will detect all the verticle lines from the image.
        verticle_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ker_len))

        # A horizontal kernel of (kernel_length x 1),
        # which will detect all the horizontal line from the image.
        hori_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (ker_len, 1))

        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ----------------- MORPHOLOGICAL OPERATIONS ------------------------ #

        # Morphological operation to detect verticle lines from an image
        # Iterations determines (somehow) how long edges can be to be detected
        # Changed from 3 to 2 to detect the small squares with the digits
        img_temp1 = cv2.erode(img_bin, verticle_ker, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_ker, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_ker, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_ker, iterations=3)

        # ---------------------- COMBINING LINES ---------------------------- #

        # Weighting parameters, this will decide the quantity
        # of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha

        # This function adds two image with specific weight parameters
        # to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha,
                                        horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255,
                                                cv2.THRESH_BINARY |
                                                cv2.THRESH_OTSU)

        if debugging:
            # Save image with vertical and horizontal lines
            cv2.imwrite(f'{dst_folder}debug_im{im_num}_'
                        f'lines.jpg', img_final_bin)

        # ----------------- FIND AND SORT CONTOURS -------------------------- #

        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(
            img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort all the contours from top to bottom.
        (contours, boundingBoxes) = sort_contours(
            contours, method="top-to-bottom")

        # ---------------------- SELECT CONTOURS ---------------------------- #
        ''' Starting from here, the code is tailored to our questionnaire. '''

        task_pdf_values = []
        overall_pdf_values = []

        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)

            # EXTRACT DENSITY FUNCTIONS
            # Din A4: 29.7cm x 21cm
            # Density function's coordinate system: 4.5cm x 4.5cm
            # Percentage of height: 29.7cm / 4.5cm = 0.15152%
            # Percentage of width: 21cm / 4.5cm = 0.21429%
            
            # Constraint 1: Only look at the right half of the image
            # Constraint 2: Ensure correct percentage of height
            # Constraint 3: Ensure correct percentage of width
            # Constraint 4: Ensure that width & height have about equal size

            if im_width * 0.5 < x \
                    and im_height * 0.20 > h > im_height * 0.10 \
                        and im_width * 0.25 > w > 0.15 \
                            and 1.08 * h > w and 1.08 * w > h:
                task_pdf_values.append([x, y, w, h])

            # OVERALL TASK PERFORMANCE
            # Din A4: 29.7cm x 21cm
            # Overall task performance rectangle: 5cm x 14.4cm
            # Percentage of height: 29.7cm / 5cm = 0.16835
            # Percentage of width: 21cm / 14.4 = 0.68571
            #   => Something is off! It is more around 0.5

            # Constraint 1: Ensure correct percentage of height
            # Constraint 2: Ensure correct percentage of width
            elif im_height * 0.2 > h > im_height * 0.1 \
                    and im_width * 0.75 > w > im_width * 0.4:
                overall_pdf_values.append([x, y, w, h])

        # ----------------- SAVE DENSITY FUNCTION CONTOURS ------------------ #

        # Iterate and check if we found contours multiple times
        # If so, save the contour which is the narrowest
        for i in range(len(task_pdf_values)):
            x, y, w, h = task_pdf_values[i]
            y_next = 0
            x_next = 0
            if i != len(task_pdf_values)-1:
                x_next, y_next, _, _ = task_pdf_values[i+1]

            # The margin in which we assign contours to the same object
            margin_h = im_height * 0.03
            margin_x = im_width * 0.03

            # Save if the distance between two contours is larger than margin.
            # Also always save the last contour. Because it is already ordered,
            # the last contour will always be smaller than the others.
            if np.abs(y - y_next) > margin_h \
                    or np.abs(x - x_next) > margin_x \
                    or i == len(task_pdf_values) - 1:
                pdf_task_counter += 1
                new_img = img[y:y + h, x:x + w]
                saving_path = f'{dst_folder}pdf_task_' \
                              f'{pdf_task_counter}.jpg'
                if not cv2.imwrite(saving_path, new_img):
                    print("Could not save image.")

        # -------------- SAVE OVERALL TASK PERFORMANCE CONTOUR -------------- #

        already_saved_overall_pdf = False
        for i in range(1, len(overall_pdf_values)):
            x, y, w, h = overall_pdf_values[i]
            y_next = 0
            if i != len(overall_pdf_values)-1:
                _, y_next, _, _ = overall_pdf_values[i+1]

            # The margin in which we assign the contour to the same object
            margin_h = im_height * 0.03

            # Save if the distance between two contours is larger than margin.
            # Also always save the last contour. Because it is already ordered,
            # the last contour will always be smaller than the others.
            if np.abs(y - y_next) > margin_h \
                    or i == len(overall_pdf_values) - 1:

                new_img = img[y:y + h, x:x + w]
                saving_path = f'{dst_folder}pdf_task1to7.jpg'

                cv2.imwrite(saving_path, new_img)

                if already_saved_overall_pdf:
                    raise RuntimeWarning(f'Found more than one overall pdf '
                                         f'contour (this shouldn\'t happen), '
                                         f'last one overwrites the others.')
                already_saved_overall_pdf = True


def extract_pdfs(subjects, subjects_folder_path, file_suffixes):
    """Extract probability density functions for all subjects.

    We iterate over all subjects, building an array containing all the paths
    to the files we want to be extracted and pass this array with a path to
    a destination folder to the extract function.

    :param subjects: list of string; the subject codes
    :param subjects_folder_path: string; path to the subject folders
    :param file_suffixes: list of string; endings of the files that should be
        read for each subject
    """
    for subject in subjects:
        dst_folder = f'{subjects_folder_path}subject_{subject}/pdfs/'

        # Array with each file we want to extract pdf's from
        pdfs_paths = []
        for fs in file_suffixes:
            pdfs_paths.append(f'{subjects_folder_path}subject_'
                              f'{subject}/raw/subject_{subject}{fs}')

        extract_pdfs_(pdfs_paths, dst_folder, debugging=False)