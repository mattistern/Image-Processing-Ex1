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
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

YIQ_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [0.59590059, -0.27455667, -0.32134392],
                         [0.21153661, -0.52273617, 0.31119955]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 311603641


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)

    if representation == LOAD_GRAY_SCALE:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        norm_img = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img
    elif representation == LOAD_RGB:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        norm_img = cv2.normalize(rgb_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm_img
    else:
        print('the representation isn\'t 1 or 2')
        return cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = cv2.imread(filename)

    if representation == LOAD_GRAY_SCALE:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
        plt.imshow(gray_img, cmap='gray')
        plt.show()
    elif representation == LOAD_RGB:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.imshow(rgb_img)
        plt.show()
    else:
        print('the representation isn\'t 1 or 2')
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)

        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    orig_shape = imgRGB.shape
    imYIQ = np.dot(imgRGB.reshape(-1, 3), YIQ_FROM_RGB.transpose()).reshape(orig_shape)
    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    orig_shape = imgYIQ.shape
    imgRGB = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(YIQ_FROM_RGB).transpose()).reshape(orig_shape)
    return imgRGB


def hist_eq(img: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    # Flattning the image and converting it into a histogram
    histOrig, bins = np.histogram(img.flatten(), bins=256, range=[0, 255])
    # Calculating the cumsum of the histogram
    cdf = histOrig.cumsum()
    # Places where cdf = 0 is ignored and the rest is stored
    # in cdf_m
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Normalizing the cdf
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    # Filling it back with zeros
    cdf = np.ma.filled(cdf_m, 0)

    # Creating the new image based on the new cdf
    imgEq = cdf[img.astype('uint8')]
    histEq, bins2 = np.histogram(imgEq.flatten(), 256, [0, 256])
    print(imgEq)
    return imgEq[:, :, 0], histOrig, histEq


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    if len(imgOrig.shape) == 2:

        img = imgOrig * 255
        imgEq, histOrig, histEq = hist_eq(img)

    else:
        img = transformRGB2YIQ(imgOrig)
        img[:, :, 0] = img[:, :, 0] * 255
        img[:, :, 0], histOrig, histEq = hist_eq(img)
        img[:, :, 0] = img[:, :, 0] / 255
        imgEq = transformYIQ2RGB(img)
    return imgEq, histOrig, histEq


def fix_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    """
        Calculate the new q using wighted average on the histogram
        :param image_hist: the histogram of the original image
        :param z: the new list of centers
        :return: the new list of wighted average
    """
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    return np.round(q).astype(int)


def fix_z(q: np.array) -> np.array:
    """
        Calculate the new z using the formula from the lecture.
        :param q: the new list of q
        :param z: the old z
        :return: the new z
    """
    z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
    z_new = np.concatenate(([0], z_new, [255]))
    return z_new


def findBestCenters(histOrig: np.ndarray, nQuant: int, nIter: int) -> (np.ndarray, np.ndarray):
    """
            Finding the best nQuant centers for quantize the image in nIter steps or when the error is minimum
            :param histOrig: hist of the image (RGB or Gray scale)
            :param nQuant: Number of colors to quantize the image to
            :param nIter: Number of optimization loops
            :return: return all centers and they color selected to build from it all the images.
        """
    Z = []
    Q = []
    # head start, all the intervals are in the same length
    z = np.arange(0, 256, round(256 / nQuant))
    z = np.append(z, [255])
    Z.append(z.copy())
    q = fix_q(z, histOrig)
    Q.append(q.copy())
    for n in range(nIter):
        z = fix_z(q)
        if (Z[-1] == z).all():  # break if nothing changed
            break
        Z.append(z.copy())
        q = fix_q(z, histOrig)
        Q.append(q.copy())
    return Z, Q


def convertToImg(imOrig: np.ndarray, histOrig: np.ndarray, yiqIm: np.ndarray, arrayQuantize: np.ndarray) -> (
        np.ndarray, float):
    """
        Executing the quantization to the original image
        :return: returning the resulting image and the MSE.
    """
    imageQ = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
    curr_hist = np.histogram(imageQ, bins=256)[0]
    err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
        imOrig.shape[0] * imOrig.shape[1])
    if len(yiqIm):  # if the original image is RGB
        yiqIm[:, :, 0] = imageQ / 255
        return transformYIQ2RGB(yiqIm), err
    return imageQ, err


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 3:
        imYIQ = transformRGB2YIQ(imOrig)
        imY = imYIQ[:, :, 0].copy()  # take only the y chanel
    else:
        imY = imOrig
    histOrig = np.histogram(imY.flatten(), bins=256)[0]
    Z, Q = findBestCenters(histOrig, nQuant, nIter)
    image_history = [imOrig.copy()]
    E = []
    for i in range(len(Z)):
        arrayQuantize = np.array([Q[i][k] for k in range(len(Q[i])) for x in range(Z[i][k], Z[i][k + 1])])
        q_img, e = convertToImg(imY, histOrig, imYIQ if len(imOrig.shape) == 3 else [], arrayQuantize)
        image_history.append(q_img)
        E.append(e)

    return image_history, E
    pass
