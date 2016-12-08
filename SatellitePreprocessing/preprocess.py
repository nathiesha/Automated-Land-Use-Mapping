#!/usr/bin/env python

import cv2
import numpy as np
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('Qt4Agg') 

img = cv2.imread('original.png')

bilateral = cv2.bilateralFilter(img,9,75,75)

cv2.imwrite('bilaterla_filtered.png',bilateral)

median = cv2.medianBlur(img,15)


img = cv2.imread('bilaterla_filtered.png')

# generating the kernels
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3)

cv2.imwrite('edge_enhancement.png',output_3)

img = cv2.imread('edge_enhancement.png')

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png',gray_image)

hist,bins = np.histogram(gray_image.flatten(),256,[0,256])
cdf = hist.cumsum()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
image_enhanced=img2
cv2.imwrite('Final.png',image_enhanced)

#recalculate cdf
hist,bins = np.histogram(image_enhanced.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()
 
plt.plot(cdf_normalized, color = 'b')
plt.hist(image_enhanced.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.savefig('histogram_final.png')
plt.show()

