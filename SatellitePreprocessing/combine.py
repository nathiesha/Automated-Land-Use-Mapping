#!/usr/bin/env python

import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('Qt4Agg') 

img1 = cv2.imread('original.png')
img2 =cv2.imread('Final.png')

combine = np.hstack((img1,img2)) #stacking images side-by-side
cv2.imwrite('combine.png',combine)


