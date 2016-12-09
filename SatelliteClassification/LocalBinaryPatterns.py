
from LBP_Module.LBP import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import argparse
import cv2
import numpy as np
import os
from osgeo import gdal
from sklearn import metrics
from scipy.misc import imsave
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import osgeo.gdal
import struct, pylab

def slidingWindow(im):
    global rgb_img
    rows,cols,sz = im.shape
    img_new = np.zeros((rows, cols, sz), dtype=np.uint8)
    s = []
    for x in range(1,rows):
        for y in range(1,cols):
            s.append(im[x-1:x+2,y-1:y+2])
            img_new[x][y][0] = 0
            img_new[x][y][1] = 0
            img_new[x][y][2] =255
    cv2.imwrite("img.tif",img_new)
    return s

def rasterizeVector(path_to_vector, cols, rows, geo_transform, projection, target_value=1):
    dataSource = gdal.OpenEx(path_to_vector, gdal.OF_VECTOR)
    layer = dataSource.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')  
    rasterDS = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    gdal.RasterizeLayer(rasterDS, [1], layer, burn_values=[target_value])
    return rasterDS

inpRaster = "/home/amanda/Desktop/lbp/image.tif"
trainData = "/home/amanda/Desktop/lbp/training"

raster_dataset = gdal.Open(inpRaster, gdal.GA_ReadOnly)
geo_transform = raster_dataset.GetGeoTransform()
proj = raster_dataset.GetProjectionRef()
bands_data = []
for b in range(1, raster_dataset.RasterCount+1):
    band = raster_dataset.GetRasterBand(b)
    bands_data.append(band.ReadAsArray())

bands_data = np.dstack(bands_data)
rows, cols, n_bands = bands_data.shape

files = [f for f in os.listdir(trainData) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(trainData, f) for f in files if f.endswith('.shp')]

labeled_pixels = []
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []


labeled_pixels = np.zeros((rows, cols))
for i, path in enumerate(shapefiles):
    label = i+1
    ds = rasterizeVector(path, cols, rows, geo_transform,
                                     proj, target_value=label)
    band = ds.GetRasterBand(1)
    ss=imsave('temp.tif',band.ReadAsArray())
    image = cv2.imread("/home/amanda/Desktop/lbp/temp.tif")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    data.append(hist)
    f1=path.split("/")[-1]
    f=f1.split('.')[0]
    labels.append(f)
    ds = None
    
classifier = RandomForestClassifier(n_jobs=4, n_estimators=10)
classifier.fit(data, labels)

img_org=cv2.imread("/home/amanda/Desktop/lbp/image.tif")
imageDataset = osgeo.gdal.Open("image.tif")
band = imageDataset.GetRasterBand(1)
rows,cols,sz = img_org.shape


# Create an empty image
img_new = np.zeros((rows, cols, sz), dtype=np.uint8)

for x in range(1,rows):
	for y in range(1,cols):
		if(x-10 > 0 and y-10 > 0):
			byteString = band.ReadRaster(x-10,y-10,x+10,y+10,20,20)
		elif(x-10 < 0 or y-10 < 0):
			byteString = band.ReadRaster(x,y,x+20,y+20,20,20)
		if(byteString):
			valueType = {osgeo.gdal.GDT_Byte: 'B', osgeo.gdal.GDT_UInt16: 'H'}[band.DataType]
			values = struct.unpack('%d%s' % (20 * 20, valueType), byteString)
			matrix = np.reshape(values, (20 , 20))
			cv2.imwrite("tmp.tif",matrix)
			image2 = cv2.imread("tmp.tif")
			gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
			hist2 = desc.describe(gray2)
			his=hist2.reshape(1, -1) 
			prediction = classifier.predict(his)[0]
			if(prediction == "A"):
				print(prediction)
				img_new[x][y][0] = 0
				img_new[x][y][1] = 0
				img_new[x][y][2] =255
			elif(prediction == "B"):
				print(prediction)
				img_new[x][y][0] = 0
				img_new[x][y][1] = 255
				img_new[x][y][2] = 0
			elif(prediction == "C"):
				print(prediction)
				img_new[x][y][0] = 255
				img_new[x][y][1] = 0
				img_new[x][y][2] = 0
		else:
			img_new[x][y][0] = 255
			img_new[x][y][1] = 255
			img_new[x][y][2] = 255

cv2.imwrite("texture.tif",img_new)


