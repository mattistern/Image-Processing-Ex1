"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import cv2
import numpy as np
from ex1_utils import LOAD_GRAY_SCALE, LOAD_RGB

#0-2 in jumps of 0.1 is 200 values
gamma_slider_max = 200
title_window = 'gamma correction'

trackbar_name = 'Alpha x %d' % gamma_slider_max
global im


"""
I took this code from:
https://docs.opencv.org/3.4/da/d6a/tutorial_trackbar.html
https://stackoverflow.com/questions/60540494/is-it-possible-to-do-partial-gamma-adjust-using-opencvs-lut
"""
def on_trackbar(val):
    #global im
    gamma = val / 100
    if gamma == 0:
        gamma = 0.1
    invGamma = 1.0 / gamma
    dst = np.array([((i / 255.0) ** invGamma) * 255
                    for i in np.arange(0, 256)]).astype("uint8")

    cv2.imshow(title_window, cv2.LUT(im, dst))

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global im
    if rep == LOAD_GRAY_SCALE:
        im = cv2.imread(img_path, 0)
    else:
        im = cv2.imread(img_path)
    cv2.namedWindow(title_window)
    name = 'Gamma x %d' % gamma_slider_max
    cv2.createTrackbar(name, title_window, 0, gamma_slider_max, on_trackbar)
    # Show some stuff
    on_trackbar(0)
    # Wait until user press some key
    cv2.waitKey()

    pass


def main():
    gammaDisplay('bac_con.png', LOAD_RGB)


if __name__ == '__main__':
    main()
