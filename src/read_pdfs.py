import numpy as np
import cv2
import os
import pathlib as pl
import matplotlib.pyplot as plt

def sort_contours(cnts, method="left-to-right"):
    """
    From the internet.
    https://www.pyimagesearch.com/2015/04/20/
    sorting-contours-using-python-and-opencv/
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def extract_pdfs(image_path_array, dst_folder, debugging=False):
    """
    Mainly from the internet.
    https://medium.com/coinmonks/a-box-detection-algorithm-
    for-any-image-containing-boxes-756c15d7ed26
    """

    pdf_task_counter = 0

    for im_num, im_path in enumerate(image_path_array):

        # Read the image in grayscale (0 == grayscale)
        img = cv2.imread(im_path, 0)

        # Thresholding the image
        (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Invert the image
        img_bin = 255 - img_bin

        if debugging:
            cv2.imwrite(f'{dst_folder}debug_im{im_num}_binary.jpg', img_bin)

        # ------------------------ KERNEL ----------------------------------- #

        # Defining a kernel length
        kernel_length = np.array(img).shape[1] // 80

        # A verticle kernel of (1 X kernel_length)
        # which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        # A horizontal kernel of (kernel_length X 1),
        # which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # ----------------- MORPHOLOGICAL OPERATIONS ------------------------ #

        # Morphological operation to detect verticle lines from an image
        # Iterations determines (somehow) how long edges can be to be detected
        # I put from 3 to 2 to detect the small squares with the digits
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
        # ---------------------- COMBINING LINES ---------------------------- #

        # Weighting parameters, this will decide the quantity
        # of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha

        # This function helps to add two image with specific weight parameter
        # to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha,
                                        horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255,
                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # FOR DEBUGGING
        # Enable this line to see verticle and horizontal
        # lines in the image which is used to find boxes
        if debugging:
            cv2.imwrite(f'{dst_folder}debug_im{im_num}_'
                        f'lines.jpg', img_final_bin)

        # ----------------- FIND AND SORT CONTOURS -------------------------- #

        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(
            img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = sort_contours(
            contours, method="top-to-bottom")

        # ------------------- ITERATE AND CUT -------------------------- #

        # TODO: Find out, why it always finds two contours
        # Because the algorithm always finds 2 times the same contour,
        # where the first one is slightly to big, we skip always the first
        skip_matching_pdf = True

        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)

            # EXTRACT PDFs
            # If width and height is geater than 80 pixel
            # and it is approximately square,
            # we save the contour.
            if 1.1 * h > w > 80 and 80 < h < 1.1 * w:
                if not skip_matching_pdf:
                    pdf_task_counter += 1
                    new_img = img[y:y + h, x:x + w]
                    if not cv2.imwrite(f'{dst_folder}pdf_task_'
                                       f'{pdf_task_counter}.jpg', new_img):
                        print("Could not save image.")
                skip_matching_pdf = not skip_matching_pdf
