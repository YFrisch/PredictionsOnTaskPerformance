try:
    import Image
except ImportError:
    from PIL import Image

import pytesseract
import os
import cv2

img = cv2.imread(f'assets/6.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f'assets/test1.png', img)


x = pytesseract.image_to_string(f'assets/test1.png', config='digits')
print(x)
print("test")
