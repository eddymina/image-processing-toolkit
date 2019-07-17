import skimage
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt 


def plot_grey(im):

    plt.imshow(im,cmap='gray')
    plt.show()


def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image

    if width is None and height is None:
        print('fsd')
        return image
        # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    return resized

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped


def contour_finder(edged):
		# find the contours in the edged image, keeping only the
	# largest ones, and initialize the screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	cnts = grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

	# loop over the contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.03 * peri, True)

		# if our approximated contour has four points, then we
		# can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# show the contour (outline) of the piece of paper
	# print("STEP 2: Find contours of paper")

	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

	return screenCnt


if __name__ == "__main__":





	import os 

	path = 'scitec_files/'
	# Function to rename multiple files 
	filename='7.jpg'


	image = cv2.imread(path+'/'+str(filename),cv2.IMREAD_GRAYSCALE)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	x,y=image.shape
	h,w=10
	plot_grey(image[y:y-h, x-w:x-w])
	image =resize(image, height=500)
	gray = cv2.GaussianBlur(image, (5, 5), 1)
	edged = cv2.Canny(gray, 75,200)
	plot_grey(edged)

	screenCnt= contour_finder(edged)
	warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	plot_grey(warped)

	print(fsdf)




	parser = argparse.ArgumentParser(description='Scanning algo',
	                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-i', '--input', type=str,
	            help='Image Path')

	parser.add_argument('-l','--level', type=int, default=141,
	                help='Level of thresholding. ')

	# parser.add_argument('-r','--rotate', type=int, default=r,
	#                 help='Level of thresholding. ')

	image = cv2.imread(parser.parse_args().input,cv2.IMREAD_GRAYSCALE)

	print("Rotating and Grayscaling... ")
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image =resize(image, height=500)
	print("Resizing... ")
	gray = cv2.GaussianBlur(image, (5, 5), 1)
	edged = cv2.Canny(gray, 75,100)
	print("Blur and Edge Detection...")


	#screenCnt= contour_finder(edged)

	#plot_grey(image)



	# apply the four point transform to obtain a top-down
	# view of the original image
	#warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
	warped = image
	print("Adjusting Image Contours... ")

	# convert the warped image to grayscale, then threshold it
	# to give it that 'black and white' paper effect

	print("Thresholding... ")

	T = skimage.filters.threshold_local(warped,parser.parse_args().level, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255

	plot_grey(warped)





