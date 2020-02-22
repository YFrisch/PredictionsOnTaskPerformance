try:
    import Image
except ImportError:
    from PIL import Image

import sys
import pytesseract
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

master_im = cv2.imread(f'assets/digits/digit_1.jpg')

# for i in range(2, 16):
#     im = cv2.imread(f'assets/digits/digit_{i}.jpg')
#     master_im = np.concatenate((master_im, im), axis=1)
#
# plt.imshow(master_im)
# plt.show()

# Read the image
img = cv2.imread(f'assets/6.jpg')

# Convert to gray scale
# If your image is not already grayscale :
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
# Pull all pixel up or down to black (0) or white (255)
# - Determine correct threshold
tresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

img = np.array(img)

delete_white_borders = True
if delete_white_borders:
    # Delete white horizontal rows
    h, w = img.shape
    new_img = []
    for i in range(h):
        im_line = img[i]
        if np.any(im_line-255):
            new_img.append(im_line)

    # Delete white vertical rows
    img = np.array(new_img).T
    h, w = img.shape
    new_img = []
    for i in range(h):
        im_line = img[i]
        if np.any(im_line-255):
            new_img.append(im_line)
    img = np.array(new_img).T

h, w = img.shape

fx = 30/h
fy = 30/h

# Resize the image if necessary
# - Put fx=1 and fy=1 for no resize
# - Put interpolation=cv2.INTER_CUBIC for upscaling
# - Put interpolation=cv2.INTER_AREA for downsampling
img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

# Thresholding again
tresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Add some white border
# Put border_size == 0 for no border
border_size = 0
img = cv2.copyMakeBorder(img, border_size, border_size,
                              border_size, border_size,
                              cv2.BORDER_CONSTANT, value=255)


plt.imshow(img, cmap='gray')
plt.show()

cv2.imwrite(f'assets/test_bin.jpg', img)

pil_img = Image.fromarray(img)
x = pytesseract.image_to_string(pil_img, config='digits')
print(x)
print("END")
