#!/usr/bin/env python
# coding: utf-8

# # Path pattern generating preprocess and initializations

# ## Libraries


import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import math


# ## Constants


def rotation_matrix(theta): 
    return np.array([[np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)]])


IMG_WIDTH = 300
IMG_HEIGHT = 300
NO_DIGITS = 10
NO_SYMBOLS = 3
PLOT_WIDTH = 15
PLOT_HEIGHT = 6
LOW_RESOLUTION_IMG_SIZE = 12
LOW_RESOLUTION_IMG_MARGIN = 1
PAPER_SIZE = 0.8 # in meters
# PAPER_CENTER = np.array([[-0.95],
#                          [-1.1]])
GLOBAL_PAPER_CENTER = np.array([[-5.8],
                                [-1.1]])
GLOBAL_ROBOT_CENTER = np.array([[-4.84],
                                [0.0]])
GLOBAL_P2R_VEC = GLOBAL_PAPER_CENTER - GLOBAL_ROBOT_CENTER
GLOBAL_PAPER_CX = GLOBAL_PAPER_CENTER[0]
GLOBAL_PAPER_CY = GLOBAL_PAPER_CENTER[1]

R2G_ANGLE = -math.pi/2 # robot to global angle
# PAPER_ROBOT_ANGLE = -math.pi
HEAD_ROTATION = rotation_matrix(R2G_ANGLE)
ROBOT_PAPER_CENTER = np.dot(HEAD_ROTATION, GLOBAL_P2R_VEC) # the coordinates in the robot frame
ROBOT_PAPER_CX = ROBOT_PAPER_CENTER[0]
ROBOT_PAPER_CY = ROBOT_PAPER_CENTER[1]


# ## Loading the set of premade images


def plot_digits(digit_img):
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    for i in range(NO_SYMBOLS):
        rs = NO_SYMBOLS/2
        col_no = int(rs)
        row_no = 2 + math.ceil(rs) - col_no # if there are remainings.
        plt.subplot(row_no, col_no, i+1)
        plt.imshow(digit_img[i],cmap='gray')
        plt.axis("off")


IMAGE_PATH = '..\\data\\other input images\\symbol'
print('initialized data is stored in the following directory:')
print('..\\data\\other input images')


number_gray = np.zeros((NO_SYMBOLS, IMG_WIDTH ,IMG_HEIGHT))
for i in range (NO_SYMBOLS):
    number_image = cv.imread(IMAGE_PATH+' '+str(i)+'.png')
    number_gray[i] = cv.cvtColor(number_image, cv.COLOR_BGR2GRAY)
    
if __name__=='__main__':
    plot_digits(number_gray)


# ## Morphology Image Processing


# Taking a matrix of size 5 as the kernel
kernel = np.ones((3,3), np.uint8)
img_dilation = np.zeros((NO_SYMBOLS, IMG_WIDTH, IMG_HEIGHT))
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
if __name__=='__main__':
    plt.figure(figsize=(12, 10))
for i in range(NO_SYMBOLS):
    img_erosion = cv.erode(number_gray[i], kernel, iterations=3)
    img_dilation[i, :, :] = cv.dilate(number_gray[i], kernel, iterations=1)
    if __name__=='__main__':
        plt.subplot(5,6, 3*i+1)
        plt.imshow(number_gray[i], cmap='gray')
        plt.title("input image", color='white')
        plt.axis("off")
        plt.subplot(5,6, 3*i+2)
        plt.imshow(img_dilation[i], cmap='gray')
        plt.title("dilated image", color='white')
        plt.axis("off")
        plt.subplot(5,6, 3*i+3)
        plt.imshow(img_erosion, cmap='gray')
        plt.title("eroded image", color='white')
        plt.axis("off")


# Taking a matrix of size 5 as the kernel
kernel1 = np.ones((5,5), np.uint8)
kernel2 = np.ones((7,7), np.uint8)
kernel3 = np.ones((3,3), np.uint8)
img_dilation1 = np.zeros((NO_SYMBOLS, IMG_WIDTH, IMG_HEIGHT))
img_dilation2 = np.zeros((NO_SYMBOLS, IMG_WIDTH, IMG_HEIGHT))
img_dilation3 = np.zeros((NO_SYMBOLS, IMG_WIDTH, IMG_HEIGHT))
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
if __name__=='__main__':
    plt.figure(figsize=(12, 10))
    
for i in range(NO_SYMBOLS):
    img_dilation1[i, :, :] = cv.dilate(number_gray[i], kernel1, iterations=1)
    img_dilation2[i, :, :] = cv.dilate(number_gray[i], kernel2, iterations=1)
    img_dilation3[i, :, :] = cv.dilate(number_gray[i], kernel3, iterations=2)
    if __name__=='__main__':
        plt.subplot(5,6, 3*i+1)
        plt.imshow(img_dilation1[i], cmap='gray')
        if(i<=1):
            plt.title("dilation a", color='white')
        plt.axis("off")
        plt.subplot(5,6, 3*i+2)
        plt.imshow(img_dilation2[i], cmap='gray')
        if(i<=1):
            plt.title("dilation b", color='white')
        plt.axis("off")
        plt.subplot(5,6, 3*i+3)
        plt.imshow(img_dilation3[i], cmap='gray')
        if(i<=1):
            plt.title("dilation c", color='white')
        plt.axis("off")


# ## Output


for i in range(NO_SYMBOLS):
    cv.imwrite(IMAGE_PATH+' '+str(i)+' dilated.png', img_dilation[i])
    
# IMAGE_PATH

