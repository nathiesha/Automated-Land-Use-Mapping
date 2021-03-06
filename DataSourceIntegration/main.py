# import the necessary packages
import csv
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.segmentation import relabel_from_one
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

no_of_buildings = 0
overlapped_buildings = 0

#longitude and latitude values of original image
y_up=6.939803836
y_down=6.931816100450768
x_low=79.89909402
x_high=79.90392199622684

til = Image.new("RGB",(300, 288))
til.save('PILImage.jpg')

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

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

cv2.imwrite('builtup.jpg',output)

# Read image
im_in = cv2.imread('builtup.jpg', cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.

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
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()

counter_red = 0
lower_blue = np.array([110, 50, 50])

# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 125

        print i

        image_mask=cv2.bitwise_or(image, image, mask = mask)
        # cv2.imshow("Applied", image_mask)
        # cv2.waitKey(0)
        cv2.imwrite('segment.png',image_mask)
        im = Image.open('segment.png')
        width,height= im.size
        pixels = im.getdata()  # get the pixels as a flattened sequence
        pix=im.load()
        white = 0
        all=0
        for pixel in pixels:
            if pixel == (255,255,255):
                white += 1
            else:
                all +=1

        print white
        print all

        if (white > 0):
                coun=0
                # cv2.imshow("Applied", image_mask)
                # cv2.waitKey(0)

                width, height = im.size
                #print width
                #print height
                diff1 = (y_up - y_down) / height
                diff2 = (x_high - x_low)  / width
                with open("refinedFSquare.csv", "rb") as inp:
                    counterr=0
                    for row in csv.reader(inp):

                        if (float(row[1]) < y_up) and (float(row[1]) > y_down) and (
                            float(row[2]) < x_high) and (
                                    float(row[2]) > x_low):
                            counterr += 1
                            print row
                            #print int(round((y_up-(float(row[1])))/diff1))
                            long = int(round((y_up-(float(row[1])))/diff1))
                            lati = int(round((float(row[2]) - x_low) / diff2))
                            #print long
                            #print lati

                            if (pix[lati,long] != (0,0,0)):
                                coun+=1
                                plt.scatter(lati, long, color='red')
                                #plt.show()
                                counter_red += 1
                                img = Image.open('segment.png')
                                img = img.convert("RGBA")

                                pixdata = img.load()

                                if(row[7]=='Transportation'):
                                    ###yellow
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (9, 249, 235, 0)
                                                pixdata[x, y] = (0, 255, 255, 0)

                                if(row[7]=='Professional'):
                                    ##light blue
                                    ###green blue
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (246, 246, 72, 0)
                                                #pink-prple pixdata[x, y] = (102, 102, 255, 0)
                                                #pixdata[x, y] = (102, 255, 255, 0)
                                                pixdata[x, y] = (255, 255, 0, 0)

                                if(row[7]=='Recreation'):
                                    ##purple
                                    ###orange
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (163, 6, 255, 0)
                                                pixdata[x, y] = (0, 165, 25, 0)

                                if(row[7]=='Hotels & Restaurants'):
                                    ##light green
                                    ###purple
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (71, 246, 89, 0)
                                                pixdata[x, y] = (0, 255, 255, 0)

                                if(row[7]=='Hospitals'):
                                    ##orange
                                    ###brown
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (105, 149, 232, 0)
                                                pixdata[x, y] = (0, 0, 125, 0)

                                if(row[7]=='Commercial & Mercantile'):
                                    ###red
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                pixdata[x, y] = (0, 0, 255, 0)

                                if(row[7]=='Education'):
                                    ###blue
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (255, 146, 1, 0)
                                                pixdata[x, y] = (255, 0, 0, 0)


                                if(row[7]=='Residential'):
                                    ##pink
                                    ###grey
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (136, 48, 223, 0)
                                                pixdata[x, y] = (125, 125, 125, 0)

                                if(row[7]=='Administration'):
                                    ###green
                                    for y in xrange(img.size[1]):
                                        for x in xrange(img.size[0]):
                                            if pixdata[x, y] != (0,0,0,255):
                                                #pixdata[x, y] = (29, 173, 12, 0)
                                                pixdata[x, y] = (0, 255, 0, 0)



                                #Image._show(img)
                                outimg = np.array(img)
                                #cv2.imshow('win',outimg)
                                #cv2.waitKey(0)
                                cv2.imwrite('image2.jpg',outimg)



                            else:
                                plt.scatter(lati, long,
                                            color='red')
                                #plt.scatter(200, 100, color='red')
                                cv2.imwrite('image1.jpg',image_mask)

                    #print counterr
                if coun>0:
                    image2 = cv2.imread('image2.jpg')
                    overlapped_buildings+=1

                else:
                    image2 = cv2.imread('image1.jpg')

                my = cv2.imread('PILImage.jpg')

                rows, cols, channels = my.shape
                roi = image2[0:rows, 0:cols]

                img2gray = cv2.cvtColor(my, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                img2_fg = cv2.bitwise_and(my, my, mask=mask)
                dst = cv2.add(img1_bg, img2_fg)
                image2[0:rows, 0:cols] = dst
                #cv2.imshow('res', image2)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                cv2.imwrite('PILImage.jpg', image2)
                no_of_buildings+=1
                #image1 = cv2.imread('image1.jpg')

imfinal=cv2.imread('PILImage.jpg')
plt.imshow( np.hstack([imfinal]))
plt.show()

print counterr
print counter_red
print no_of_buildings
print overlapped_buildings


