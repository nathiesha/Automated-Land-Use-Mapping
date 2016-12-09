import numpy as np
import os
from osgeo import gdal
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

def rasterizeVector(path_to_vector,cols,rows,geo_transform,projection):
	lblRaster=np.zeros((rows, cols))
	for i, path in enumerate(path_to_vector):
		label = i+1
		# open the input datasource and read content
		inputDS = gdal.OpenEx(path, gdal.OF_VECTOR)
		shpLayer = inputDS.GetLayer(0)
		# Create the destination data source
		driver = gdal.GetDriverByName('MEM') 
		rasterDS = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
		# Define spatial reference
		rasterDS.SetGeoTransform(geo_transform)
		rasterDS.SetProjection(projection)
		# Rasterize
		gdal.RasterizeLayer(rasterDS, [1], shpLayer, burn_values=[label])
		# Get a raster band
		rBand = rasterDS.GetRasterBand(1)
		lblRaster += rBand.ReadAsArray()
		rasterDS = None
	return lblRaster

def createGeotiff(outRaster, data, geo_transform, projection):
    # Create a GeoTIFF file with the given data
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    rasterDS = driver.Create(outRaster, cols, rows, 1, gdal.GDT_Byte)
    rasterDS.SetGeoTransform(geo_transform)
    rasterDS.SetProjection(projection)
    band = rasterDS.GetRasterBand(1)
    band.WriteArray(data)
    dataset = None

inpRaster = "/home/amanda/Desktop/km/google_satellite_imagery/input_raster2.tif"
outRaster = "/home/amanda/Desktop/rr_classification.tiff"
trainData = "/home/amanda/Desktop/km/google_satellite_imagery/train"
testData = "/home/amanda/Desktop/km/google_satellite_imagery/train"

# Open raster dataset
rasterDS = gdal.Open(inpRaster, gdal.GA_ReadOnly)
# Get spatial reference
geo_transform = rasterDS.GetGeoTransform()
projection = rasterDS.GetProjectionRef()

# Extract band's data and transform into a numpy array
bandsData = []
for b in range(1, rasterDS.RasterCount+1):
    band = rasterDS.GetRasterBand(b)
    bandsData.append(band.ReadAsArray())
bandsData = np.dstack(bandsData)
rows, cols, noBands = bandsData.shape

# Read vector data, and rasterize all the vectors in the given directory into a single labelled raster
files = [f for f in os.listdir(trainData) if f.endswith('.shp')]
classes = [f.split('.')[0] for f in files]
shapefiles = [os.path.join(trainData, f) for f in files if f.endswith('.shp')]
lblRaster = rasterizeVector(shapefiles, rows, cols, geo_transform, projection)

# Prepare training data (set of pixels used for training) and labels
isTrain = np.nonzero(lblRaster)
trainingLabels = lblRaster [isTrain]
trainingData = bandsData[isTrain]

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_jobs=4, n_estimators=10)
classifier.fit(trainingData, trainingLabels)

# Predict class label of unknown pixels
noSamples = rows*cols
flat_pixels = bandsData.reshape((noSamples, noBands))
result = classifier.predict(flat_pixels)
classification = result.reshape((rows, cols))

# Create a GeoTIFF file with the given data
createGeotiff(outRaster, classification, geo_transform, projection)


# Test classification accuracy
shapefiles = [os.path.join(testData, "%s.shp"%c) for c in classes]
verificationPixels = rasterizeVector(shapefiles, rows, cols, geo_transform, projection)
forVerification = np.nonzero(verificationPixels)
verificationLabels = verificationPixels[forVerification]
predictedLabels = classification[forVerification]

print("Confussion matrix:\n%s" % metrics.confusion_matrix(verificationLabels, predictedLabels))
targetNames = ['Class %s' % s for s in classes]
print("Classification report:\n%s" % metrics.classification_report(verificationLabels, predictedLabels, target_names=targetNames))
print("Classification accuracy: %f" % metrics.accuracy_score(verificationLabels, predictedLabels))

