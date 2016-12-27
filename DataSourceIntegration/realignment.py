# import the necessary packages
import csv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from math import *
import math

def haversine(lon1, lat1, lon2, lat2):
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
	dlon = lon2 - lon1
	dlat = lat2 - lat1
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a))
	km = 6367 * c
	#convert from km to meters
	meters = km * 1000
        return round(meters)

# the threshold is set to 20 meters
radius=20

no_of_buildings = 0
overlapped_buildings = 0
x_left=0
x_right=0
y_low=0
y_high=0

overlap=0

#longitude and latitude values of original image
y_up=6.966753172
y_down=6.965155727
x_low=79.8811278
x_high=79.88595578

#1-6.975736284,6.9741388699832125,80.20451983,80.20934780622675
#2-6.966753172,6.965155727,79.8811278,79.88595578

havesine_distance_x1= haversine(x_low,y_down,x_high,y_down)
havesine_distance_x2= haversine(x_low,y_up,x_high,y_up)
havesine_distance_y1= haversine(x_low,y_down,x_low,y_up)
havesine_distance_y2= haversine(x_high,y_down,x_high,y_up)

havesine_x=(havesine_distance_x1+havesine_distance_x2)/2
havesine_y=(havesine_distance_y1+havesine_distance_y2)/2

point_dist=[]

# load the image
image = cv2.imread('original0.jpg', cv2.IMREAD_COLOR)

#green boundary
# define the list of boundaries
# boundaries = [
# 	([100, 200, 100], [150, 255, 150])
# ]

#red boundary
# # define the list of boundaries
# boundaries = [([0, 0, 100], [30, 30, 180])]

#blue boundary
# define the list of boundaries
boundaries = [
	([90, 0, 0], [150, 15, 15])
]

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
# fig = plt.figure("Superpixels")
# ax = fig.add_subplot(1, 1, 1)
#ax.imwrite(,mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
# plt.axis("off")
# plt.show()

counter_red = 0
lower_blue = np.array([110, 50, 50])
result=[]

imo = Image.open('original0.jpg')
widtho, heighto = imo.size

til = Image.new("RGB",(widtho, heighto))
til.save('PILImage.jpg')

diff_x=widtho/havesine_x
diff_y=heighto/havesine_y
diff_avg=(diff_x+diff_y)/2
threshold_pixels=round(diff_avg*radius)



##--------------------------Foursquare Plotting--------------------------------------------------

points=[]
diff1 = (y_up - y_down) / heighto
diff2 = (x_high - x_low) / widtho

til = Image.new("RGB",(widtho, heighto))
til.save('plotImage.jpg')

plotIm = plt.imread('plotImage.jpg')

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

# plt.imshow( np.hstack([plotIm]))
# plt.show()

##----------------------------------------------------------------------------------------------------


# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 125

        # print i

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
            # cv2.imshow("Applied", image_mask)
            # cv2.waitKey(0)
##------------------------------------save mid point of buildings--------------------------------------------------
            x_left = 0
            x_right = 0
            y_low = 0
            y_high = 0

            for x in range(width):
                for y in range(height):
                    if pix[x,y] == (255, 255, 255):
                        if set_val == 0:
                            x_left=x
                            y_low=y
                            y_high=y
                            # print (x,y)
                            set_val=1

                        if set_val == 1:
                            x_right = x
                            # print (x,y)

                        if y<y_low:
                            y_low=y

                        if y>y_high:
                            y_high=y


            # print x_left
            # print x_right
            # print y_low
            # print  y_high

            x_avg=x_left+(x_right-x_left)/2
            y_avg=y_low+(y_high-y_low)/2

            # print (x_avg,y_avg)

            result.append((i,(x_avg,y_avg)))


            cv2.imwrite('segment' + str(i) + '.png', image_mask)
            im = Image.open('segment' + str(i) + '.png')
##---------------------------------------Find overlapped buildings and save them---------------------------------------------------------------------------------------------------
            point_no=0
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
                    print 'hey'

                    if(type=='Transportation'):
                        ###yellow
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (9, 249, 235, 0)
                                    pixdata[x, y] = (0, 255, 255, 0)

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
                        ###orange
                        for y in xrange(img.size[1]):
                            for x in xrange(img.size[0]):
                                if pixdata[x, y] != (0,0,0,255):
                                    #pixdata[x, y] = (163, 6, 255, 0)
                                    pixdata[x, y] = (0, 165, 25, 0)

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

                    #Image._show(img)
                    outimg = np.array(img)
                    cv2.imwrite('color'+str(overlap)+'.jpg',outimg)
                    # cv2.imshow('win',outimg)
                    # cv2.waitKey(0)
                    #cv2.imwrite('image2.jpg',outimg)
                    points.remove(point)

 ##--------------------------------------else merge all the white segments to one image-----------------------------------------------
               else:

                   Euclidean_distance=math.sqrt(((x_avg-latitude)*(x_avg-latitude))+((y_avg-longitude)*(y_avg-longitude)))

                   if Euclidean_distance<=threshold_pixels:
                       point_dist.append((latitude,longitude,type,i,Euclidean_distance))

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

               point_no+=1
# cv2.imshow('res', img1)
# cv2.waitKey(0)

print 'no of buildings'
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

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
    ##-------------------------------------------------------------------------------------------------------
###----------------------------------Realignment of other Foursquare points---------------------------------
lowest=radius
low_num=0
low_set=[]
for (k,l) in enumerate(points):
    lowest = radius
    for (i, j) in enumerate(point_dist):
        a,b,c,d,e=j
        f,g,h=l

        if (a==f and b==g):
            if (e<lowest):
                lowest=e
                low_num=d

    low_set.append((l,lowest,low_num))

print 'realigned foursquare points:'

for (m,n) in enumerate(low_set):

            row,dist,num=n
            print row
            a,b,type=row
            img = Image.open('segment'+str(num)+'.png')
            img = img.convert("RGBA")

            pixdata = img.load()

            if (type == 'Transportation'):
                ###yellow
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (9, 249, 235, 0)
                            pixdata[x, y] = (0, 255, 255, 0)

            if (type == 'Professional'):
                ##light blue
                ###green blue
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (246, 246, 72, 0)
                            # pink-prple pixdata[x, y] = (102, 102, 255, 0)
                            # pixdata[x, y] = (102, 255, 255, 0)
                            pixdata[x, y] = (255, 255, 0, 0)

            if (type == 'Recreation'):
                ##purple
                ###orange
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (163, 6, 255, 0)
                            pixdata[x, y] = (0, 165, 25, 0)

            if (type == 'Hotels & Restaurants'):
                ##light green
                ###purple
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (71, 246, 89, 0)
                            pixdata[x, y] = (0, 255, 255, 0)

            if (type == 'Hospitals'):
                ##orange
                ###brown
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (105, 149, 232, 0)
                            pixdata[x, y] = (0, 0, 125, 0)

            if (type == 'Commercial & Mercantile'):
                ###red
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            pixdata[x, y] = (0, 0, 255, 0)


            if (type == 'Education'):
                ###blue
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (255, 146, 1, 0)
                            pixdata[x, y] = (255, 0, 0, 0)

            if (type == 'Residential'):
                ##pink
                ###grey
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (136, 48, 223, 0)
                            pixdata[x, y] = (125, 125, 125, 0)

            if (type == 'Administration'):
                ###green
                for y in xrange(img.size[1]):
                    for x in xrange(img.size[0]):
                        if pixdata[x, y] != (0, 0, 0, 255):
                            # pixdata[x, y] = (29, 173, 12, 0)
                            pixdata[x, y] = (0, 255, 0, 0)

            # Image._show(img)
            outimg = np.array(img)
            cv2.imwrite('color' + str(overlap+1) + '.jpg', outimg)

            # Load two images
            img1 = cv2.imread('PILImage.jpg')
            img2 = cv2.imread('color' + str(overlap+1) + '.jpg')

            # create a ROI
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
            overlap+=1

plotIm=plt.imread('PILImage.jpg')
plt.imshow( np.hstack([plotIm]))
plt.show()

cv2.imshow('Final visualization', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

print 'no of point overlapped with buildings:'
print overlap

print 'no of points realigned:'
print len(low_set)

