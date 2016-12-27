# import the necessary packages
import csv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

no_of_buildings = 0
overlapped_buildings = 0
overlap=0

#longitude and latitude values of original image
y_up=6.939803836
y_down=6.931816100450768
x_low=79.89909402
x_high=79.90392199622684


# load the image
image = cv2.imread('original0.jpg', cv2.IMREAD_COLOR)

#green boundary
# define the list of boundaries
boundaries = [
	([100, 200, 100], [150, 255, 150])
]

#red boundary
# define the list of boundaries
#boundaries = [([0, 0, 100], [30, 30, 180])]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('builtup.jpg',output)

# Read image
im_in = cv2.imread('builtup.jpg', cv2.IMREAD_GRAYSCALE);

# Threshold.
th, im_th = cv2.threshold(im_in, 10, 200, cv2.THRESH_BINARY_INV);

# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

thresh = cv2.threshold(im_floodfill_inv, 177, 255, cv2.THRESH_BINARY)[1]

# Display images.
cv2.imwrite('grey.jpg',thresh)
# cv2.imshow("Floodfilled Image", im_out)
# cv2.waitKey(0)


# load the image and apply SLIC and extract (approximately)
# the supplied number of segments
image = cv2.imread('grey.jpg')
segments = slic(img_as_float(image), n_segments = 500, compactness=10,sigma = 5)

# show the output of SLIC
fig = plt.figure("segmented image")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

counter_red = 0
lower_blue = np.array([110, 50, 50])
result=[]

imo = Image.open('original0.jpg')
widtho, heighto = imo.size

til = Image.new("RGB",(widtho, heighto))
til.save('PILImage.jpg')

##--------------------------Foursquare Plotting--------------------------------------------------

points=[]
diff1 = (y_up - y_down) / heighto
diff2 = (x_high - x_low) / widtho

til = Image.new("RGB",(widtho, heighto))
til.save('plotImage.jpg')

plotIm = plt.imread('plotImage.jpg')

print 'Foursquare points found in the region:'

with open("refinedFSquare.csv", "rb") as inp:
    for row in csv.reader(inp):

        if (float(row[1]) < y_up) and (float(row[1]) > y_down) and (
                    float(row[2]) < x_high) and (
                    float(row[2]) > x_low):
            print row
            # print int(round((y_up-(float(row[1])))/diff1))
            long = int(round((y_up - (float(row[1]))) / diff1))
            lati = int(round((float(row[2]) - x_low) / diff2))
            points.append((lati,long,row[7]))
            # print long
            # print lati
            plt.scatter([lati], [long], color='red')

plt.imshow( np.hstack([plotIm]))
plt.show()

##----------------------------------------------------------------------------------------------------


# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 125

        set_val=0
        image_mask=cv2.bitwise_or(image, image, mask = mask)
        #cv2.imshow("Applied", image_mask)
        #cv2.waitKey(0)
        cv2.imwrite('segment.png',image_mask)
        im = Image.open('segment.png')
        width,height= im.size
        # print width
        # print height
        pixels = im.getdata()  # get the pixels as a flattened sequence
        pix = im.load()
        pix=im.load()
        white = 0
        all=0
##------------------------------Identify only buildings out of selected segements----------------------------------------------

        for pixel in pixels:
            if pixel == (255,255,255):
                white += 1
            else:
                all +=1

            # print white
            # print all

        if (white > 0):
            coun=0
            cv2.imwrite('segment' + str(i) + '.png', image_mask)
            im = Image.open('segment' + str(i) + '.png')
##---------------------------------------Find overlapped buildings and save them---------------------------------------------------------------------------------------------------

            for point in points:
               latitude, longitude, type=point
               if (pix[latitude,longitude] != (0,0,0)):
                    overlap=overlap+1
                    # plt.scatter(lati, long, color='red')
                    # #plt.show()
                    # counter_red += 1
                    img = Image.open('segment.png')
                    img = img.convert("RGBA")

                    pixdata = img.load()

                    if(type=='Transportation'):
                        ###yellow
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (9, 249, 235, 0)
                                    pixdata[x, y] = (0, 0, 255, 0)

                    if (type == 'Professional'):
                        ##light blue
                        ###green blue
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (246, 246, 72, 0)
                                    #pink-prple pixdata[x, y] = (102, 102, 255, 0)
                                    #pixdata[x, y] = (102, 255, 255, 0)
                                    pixdata[x, y] = (255, 255, 0, 0)

                    if(type=='Recreation'):
                        ##purple
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (163, 6, 255, 0)
                                    pixdata[x, y] = (0, 255, 255, 0)

                    if(type=='Hotels & Restaurants'):
                        ##light green
                        ###purple
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (71, 246, 89, 0)
                                    pixdata[x, y] = (0, 255, 255, 0)

                    if(type=='Hospitals'):
                        ##orange
                        ###brown
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (105, 149, 232, 0)
                                    pixdata[x, y] = (0, 0, 125, 0)

                    if(type=='Commercial & Mercantile'):
                        ###red
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    pixdata[x, y] = (0, 0, 255, 0)

                    if(type=='Education'):
                        ###blue
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                #pixdata[x, y] = (255, 146, 1, 0)
                                    pixdata[x, y] = (255, 0, 0, 0)


                    if(type=='Residential'):
                        ##pink
                        ###grey
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (136, 48, 223, 0)
                                    pixdata[x, y] = (125, 125, 125, 0)

                    if(type=='Administration'):
                        ###green
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (29, 173, 12, 0)
                                    pixdata[x, y] = (0, 255, 0, 0)

                    outimg = np.array(img)
                    cv2.imwrite('color'+str(overlap)+'.jpg',outimg)
                    # cv2.imshow('win',outimg)
                    # cv2.waitKey(0)

 ##--------------------------------------else merge all the white segments to one image-----------------------------------------------
               else:

                   # Load two images
                   img1 = cv2.imread('PILImage.jpg')
                   img2 = cv2.imread('segment'+str(i)+'.png')

                   # I want to put logo on top-left corner, So I create a ROI
                   rows, cols, channels = img2.shape
                   roi = img1[0:rows, 0:cols]

                   # Now create a mask of logo and create its inverse mask also
                   img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                   ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                   mask_inv = cv2.bitwise_not(mask)

                   # Now black-out the area of logo in ROI
                   img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                   # Take only region of logo from logo image.
                   img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

                   # Put logo in ROI and modify the main image
                   dst = cv2.add(img1_bg, img2_fg)
                   img1[0:rows, 0:cols] = dst
                   cv2.destroyAllWindows()

                   cv2.imwrite('PILImage.jpg', img1)
                   no_of_buildings+=1
                   #image1 = cv2.imread('image1.jpg')

# cv2.imshow('res', img1)
# cv2.waitKey(0)
print 'no of building blocks identified:'
print i

##-----------------------------------------Merge the coloured segments---------------------------------------------
for i in range (0,overlap):
    # Load two images
    img1 = cv2.imread('PILImage.jpg')
    img2 = cv2.imread('color'+str(i+1)+'.jpg')

    #create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst


    cv2.imwrite('PILImage.jpg', img1)

plotIm=plt.imread('PILImage.jpg')
plt.imshow( np.hstack([plotIm]))
plt.show()
cv2.imshow('Final visualization', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
    ##-------------------------------------------------------------------------------------------------------
print 'no of point overlapped with buildings:'
print overlap
# print counter_red
# print no_of_buildings
# print overlapped_buildings
