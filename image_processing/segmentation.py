# -- segmentation -- 

# 	otsu(gray,plot=False):

# 	remove_background_noise(img, dilations=1, kernel_size=5, thresh=174)

# 	kmeans_segmenter(img,clusters)

# 	watershed

# 	grab_contours(cnts)

# 	order_points(pts):

# 	four_point_transform(image, pts):

from image_processing.im_processing import cut
import numpy as np
from sklearn.cluster import KMeans
import cv2 


def otsu(gray,plot=False):
	pixel_number = gray.shape[0] * gray.shape[1]
	mean_weight = 1.0/pixel_number
	his, bins = np.histogram(gray, np.arange(0,257))
	final_thresh = -1
	final_value = -1
	intensity_arr = np.arange(256)

	threshes= []
	scatter = []

	for t in bins[3:-3]: # This goes through all pixel values 

	    pcb = np.sum(his[:t]) #sum of histogram vals < thresh
	    pcf = np.sum(his[t:]) #sum of histogram vals > thresh
	    Wb = pcb * mean_weight #weighted background 
	    Wf = pcf * mean_weight #weighted foreground 

	    if pcf != 0 and pcb !=0:

		    mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb) #mean = [pixel_val * his_val / sum of pixels] < thresh
		    muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf) #mean = [pixel_val * his_val / sum of pixels] > thresh

		    value = Wb * Wf * (mub - muf) ** 2 #within class variance

		    threshes.append(t)
		    scatter.append(value)
		    if value > final_value:
		        final_thresh = t
		        final_value = value

	if plot != False:
		
		plt.scatter(threshes,scatter)
		print(max(scatter))
		plt.axhline(max(scatter),c='r')
		best_thresh = final_thresh
		
		plt.axvline(best_thresh ,c='r')
		plt.title('Optimal Threshold: {}'.format(best_thresh))
		plt.xlabel('Thresholds')
		plt.ylabel('Between Class Scatter')
		plt.show()
	final_img = gray.copy()
	return final_thresh,cut(final_img,final_thresh)

def kmeans_segmenter(img,clusters):
	img_norm = img/255.0
	if len(img.shape)==3:
		img_norm=img_norm.reshape(img.shape[0]*img.shape[1], img.shape[2])


	km= KMeans(n_clusters=clusters, random_state=0).fit(img_norm)
	seg_im= km.cluster_centers_[km.labels_]
	if len(img.shape)==3:
		return seg_im.reshape(img.shape[0],img.shape[1], img.shape[2])
	else:
		return seg_im

# def grab_contours(cnts):
#     # if the length the contours tuple returned by cv2.findContours
#     # is '2' then we are using either OpenCV v2.4, v4-beta, or
#     # v4-official
#     if len(cnts) == 2:
#         cnts = cnts[0]

#     # if the length of the contours tuple is '3' then we are using
#     # either OpenCV v3, v4-pre, or v4-alpha
#     elif len(cnts) == 3:
#         cnts = cnts[1]

#     # otherwise OpenCV has changed their cv2.findContours return
#     # signature yet again and I have no idea WTH is going on
#     else:
#         raise Exception(("Contours tuple must have length 2 or 3, "
#             "otherwise OpenCV changed their cv2.findContours return "
#             "signature yet again. Refer to OpenCV's documentation "
#             "in that case"))

#     # return the actual contours array
#     return cnts

# def order_points(pts):
# 	# initialzie a list of coordinates that will be ordered
# 	# such that the first entry in the list is the top-left,
# 	# the second entry is the top-right, the third is the
# 	# bottom-right, and the fourth is the bottom-left
# 	rect = np.zeros((4, 2), dtype = "float32")
 
# 	# the top-left point will have the smallest sum, whereas
# 	# the bottom-right point will have the largest sum
# 	s = pts.sum(axis = 1)
# 	rect[0] = pts[np.argmin(s)]
# 	rect[2] = pts[np.argmax(s)]
 
# 	# now, compute the difference between the points, the
# 	# top-right point will have the smallest difference,
# 	# whereas the bottom-left will have the largest difference
# 	diff = np.diff(pts, axis = 1)
# 	rect[1] = pts[np.argmin(diff)]
# 	rect[3] = pts[np.argmax(diff)]
 
# 	# return the ordered coordinates
# 	return rect

# def four_point_transform(image, pts):
# 	# obtain a consistent order of the points and unpack them
# 	# individually
# 	rect = order_points(pts)
# 	(tl, tr, br, bl) = rect
 
# 	# compute the width of the new image, which will be the
# 	# maximum distance between bottom-right and bottom-left
# 	# x-coordiates or the top-right and top-left x-coordinates
# 	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
# 	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
# 	maxWidth = max(int(widthA), int(widthB))
 
# 	# compute the height of the new image, which will be the
# 	# maximum distance between the top-right and bottom-right
# 	# y-coordinates or the top-left and bottom-left y-coordinates
# 	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
# 	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
# 	maxHeight = max(int(heightA), int(heightB))
 
# 	# now that we have the dimensions of the new image, construct
# 	# the set of destination points to obtain a "birds eye view",
# 	# (i.e. top-down view) of the image, again specifying points
# 	# in the top-left, top-right, bottom-right, and bottom-left
# 	# order
# 	dst = np.array([
# 		[0, 0],
# 		[maxWidth - 1, 0],
# 		[maxWidth - 1, maxHeight - 1],
# 		[0, maxHeight - 1]], dtype = "float32")
 
# 	# compute the perspective transform matrix and then apply it
# 	M = cv2.getPerspectiveTransform(rect, dst)
# 	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
# 	# return the warped image
# 	return warped


# def contour_finder(edged):
# 		# find the contours in the edged image, keeping only the
# 	# largest ones, and initialize the screen contour
# 	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 	cnts = grab_contours(cnts)
# 	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# 	# loop over the contours
# 	for c in cnts:
# 		# approximate the contour
# 		peri = cv2.arcLength(c, True)
# 		approx = cv2.approxPolyDP(c, 0.03 * peri, True)

# 		# if our approximated contour has four points, then we
# 		# can assume that we have found our screen
# 		if len(approx) == 4:
# 			screenCnt = approx
# 			break

# 	# show the contour (outline) of the piece of paper
# 	# print("STEP 2: Find contours of paper")

# 	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

# 	return screenCnt


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


#function to transform image to four points
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)

    # # multiply the rectangle by the original ratio
    # rect *= ratio

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
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


#function to find two largest countours which ones are may be
#  full image and our rectangle edged object
def findLargestCountours(cntList, cntWidths):
    newCntList = []
    newCntWidths = []

    #finding 1st largest rectangle
    first_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[first_largest_cnt_pos])
    newCntWidths.append(cntWidths[first_largest_cnt_pos])

    #removing it from old
    cntList.pop(first_largest_cnt_pos)
    cntWidths.pop(first_largest_cnt_pos)

    #finding second largest rectangle
    seccond_largest_cnt_pos = cntWidths.index(max(cntWidths))

    # adding it in new
    newCntList.append(cntList[seccond_largest_cnt_pos])
    newCntWidths.append(cntWidths[seccond_largest_cnt_pos])

    #removing it from old
    cntList.pop(seccond_largest_cnt_pos)
    cntWidths.pop(seccond_largest_cnt_pos)

    print('Old Screen Dimentions filtered', cntWidths)
    print('Screen Dimentions filtered', newCntWidths)
    return newCntList, newCntWidths


#driver function which identifieng 4 corners and doing four point transformation
def convert_object(image, screen_size = None, isDebug = False):

    # image = imutils.resize(image, height=300)
    # ratio = image.shape[0] / 300.0


    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # 11  //TODO 11 FRO OFFLINE MAY NEED TO TUNE TO 5 FOR ONLINE

    gray = cv2.medianBlur(gray, 5)
    edged = cv2.Canny(gray, 30, 400)

    if isDebug  : cv2.imshow('edged', edged)
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour

    _, countours, hierarcy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if isDebug : print('length of countours ', len(countours))

    imageCopy = image.copy()
    if isDebug : cv2.imshow('drawn countours', cv2.drawContours(imageCopy, countours, -1, (0, 255, 0), 1))


    # approximate the contour
    cnts = sorted(countours, key=cv2.contourArea, reverse=True)
    screenCntList = []
    scrWidths = []
    for cnt in cnts:
        peri = cv2.arcLength(cnt, True)  # cnts[1] always rectangle O.o
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        screenCnt = approx
        # print(len(approx))

        if (len(screenCnt) == 4):

            (X, Y, W, H) = cv2.boundingRect(cnt)
            # print('X Y W H', (X, Y, W, H))
            screenCntList.append(screenCnt)
            scrWidths.append(W)

        # else:
        #     print("4 points not found")

    print('Screens found :', len(screenCntList))
    print('Screen Dimentions', scrWidths)

    screenCntList, scrWidths = findLargestCountours(screenCntList, scrWidths)

    if not len(screenCntList) >=2: #there is no rectangle found
        return None
    elif scrWidths[0] != scrWidths[1]: #mismatch in rect
        return None

    if isDebug :   cv2.imshow(" Screen", cv2.drawContours(image.copy(), [screenCntList[0]], -1, (0, 255, 0), 3))

    # now that we have our screen contour, we need to determine
    # the top-left, top-right, bottom-right, and bottom-left
    # points so that we can later warp the image -- we'll start
    # by reshaping our contour to be our finals and initializing
    # our output rectangle in top-left, top-right, bottom-right,
    # and bottom-left order
    pts = screenCntList[0].reshape(4, 2)
    print('Found bill rectagle at ', pts)
    rect = order_points(pts)
    print(rect)

    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped = four_point_transform(image, pts)

    return warped

    # convert the warped image to grayscale and then adjust
    # the intensity of the pixels to have minimum and maximum
    # values of 0 and 255, respectively
    warp = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warp = exposure.rescale_intensity(warp, out_range=(0, 255))


    # # show the original and warped images
    # if(isDebug):
    #     cv2.imshow("Original", image)
    #     cv2.imshow("warp", warp)
    #     cv2.waitKey(0)

    # if(screen_size != None):
    #     return cv2.cvtColor(cv2.resize(warp, screen_size), cv2.COLOR_GRAY2RGB)
    # else:
    #     return cv2.cvtColor(warp, cv2.COLOR_GRAY2RGB)
def neighbourhood(image, x, y):
    # Save the neighbourhood pixel's values in a dictionary
    neighbour_region_numbers = {}
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i == 0 and j == 0):
                continue
            if (x+i < 0 or y+j < 0): # If coordinates out of image range, skip
                continue
            if (x+i >= image.shape[0] or y+j >= image.shape[1]): # If coordinates out of image range, skip
                continue
            if (neighbour_region_numbers.get(image[x+i][y+j]) == None):
                neighbour_region_numbers[image[x+i][y+j]] = 1 # Create entry in dictionary if not already present
            else:
                neighbour_region_numbers[image[x+i][y+j]] += 1 # Increase count in dictionary if already present

    # Remove the key - 0 if exists
    if (neighbour_region_numbers.get(0) != None):
        del neighbour_region_numbers[0]

    # Get the keys of the dictionary
    keys = list(neighbour_region_numbers)

    # Sort the keys for ease of checking
    keys.sort()

    if (keys[0] == -1):
        if (len(keys) == 1): # Separate region
            return -1
        elif (len(keys) == 2): # Part of another region
            return keys[1]
        else: # Watershed
            return 0
    else:
        if (len(keys) == 1): # Part of another region
            return keys[0]
        else: # Watershed
            return 0

def watershed_segmentation(image):
    # Create a list of pixel intensities along with their coordinates
    intensity_list = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            # Append the tuple (pixel_intensity, xy-coord) to the end of the list
            intensity_list.append((image[x][y], (x, y)))

    # Sort the list with respect to their pixel intensities, in ascending order
    intensity_list.sort()

    # Create an empty segmented_image numpy ndarray initialized to -1's
    segmented_image = numpy.full(image.shape, -1, dtype=int)

    # Iterate the intensity_list in ascending order and update the segmented image
    region_number = 0
    for i in range(len(intensity_list)):
        # Print iteration number in terminal for clarity

        # Get the pixel intensity and the x,y coordinates
        intensity = intensity_list[i][0]
        x = intensity_list[i][1][0]
        y = intensity_list[i][1][1]

        # Get the region number of the current pixel's region by checking its neighbouring pixels
        region_status = neighbourhood(segmented_image, x, y)

        # Assign region number (or) watershed accordingly, at pixel (x, y) of the segmented image
        if (region_status == -1): # Separate region
            region_number += 1
            segmented_image[x][y] = region_number
        elif (region_status == 0): # Watershed
            segmented_image[x][y] = 0
        else: # Part of another region
            segmented_image[x][y] = region_status

    # Return the segmented image
    return segmented_image

