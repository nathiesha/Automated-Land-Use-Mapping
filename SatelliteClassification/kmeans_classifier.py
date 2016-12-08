import numpy as np
import gdal, gdalconst
from scipy.cluster.vq import *
from scipy.misc import imsave
import matplotlib.pyplot as plt

# Input raster location
input_raster="dnn.tif"

# Opening GDAL supported raster datastore
input_dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)

# Loop through all raster bands
bands_dataset = []
for b in range(1, input_dataset.RasterCount+1):
    band = input_dataset.GetRasterBand(b)
    bands_dataset.append(band.ReadAsArray())

# Stack 2D arrays (image) into a single 3D array
stack = np.dstack(bands_dataset)

# Get the dimensions of 'stack' in (no. of rows, no. of columns, no. of bands) format
rows, cols, n_bands = stack.shape

# Total number of data samples in stack
n_samples = rows*cols

# Flatten stack to rows
stack_flat = stack.reshape((n_samples, n_bands))

# Apply k-means clustering
n_clusters=4
centroids, variance = kmeans(stack_flat,n_clusters)
code, distance = vq(stack_flat, centroids)
cluster_img = code.reshape(stack.shape[0], stack.shape[1])

# Save clustered image 
imsave('frr.tif',cluster_img)

# Visualization

# Add original raster to first sub plot
ax = plt.subplot(241)
plt.axis('off')
ax.set_title('Original Image')
plt.imshow(stack)

# Display clustered images for different number of clusters
for i in range(7):

 centroids, variance = kmeans(stack_flat, i+2)
 code, distance = vq(stack_flat, centroids)
 cluster_img = code.reshape(stack.shape[0], stack.shape[1])
   
 ax = plt.subplot(2,4,i+2)
 plt.axis('off')
 xlabel = str(i+2) , ' clusters'
 ax.set_title(xlabel)
 plt.imshow(cluster_img)
    
plt.show()

