import numpy as np
import cv2
from ex1_utils import imReadAndConvert, imDisplay, transformYIQ2RGB, transformRGB2YIQ, hsitogramEqualize, quantizeImage
from typing import List
from PIL import Image
from matplotlib import pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

if __name__ == '__main__':
    # imDisplay("beach.jpg", 5)
    # imEq, histOrig, histEq = hsitogramEqualize(imReadAndConvert("bac_con.png", 3))
    #
    # plt.imshow(imReadAndConvert("bac_con.png", 3))
    # plt.show()
    # plt.imshow(imEq)
    # plt.show()
    # plt.hist(histOrig)
    # plt.show()

    qImage_i_list, err_i_list = quantizeImage(imReadAndConvert("bac_con.png", 3), 3, 2)
    print('qImage_i_list: ', qImage_i_list)
    plt.imshow(qImage_i_list[0])
    plt.show()
