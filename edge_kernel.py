# deps:
# conda install opencv

import cv2
import numpy as np
import matplotlib.pyplot as plt

# reading material :
# opencv Canny() function desc
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html

# inputs
img_path = "img/img1.png"

def edge_dec(path):
    img = cv2.imread(img_path, 0)
    edges = cv2.Canny(img, 0, 300)

    plt.subplot(121),plt.imshow(img, cmap = "gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges, cmap = "gray")
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

edge_dec(img_path)
