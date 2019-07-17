import im_processing as imp
import mahotas as mh 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy
import sys 


import matplotlib.cm as cm
from scipy import misc
from scipy import stats
import scipy.spatial.distance as dist

gray= cv.imread('endo_cells2.0.jpg',cv.IMREAD_GRAYSCALE )



def gaussian_filter(img, kernel_size= 3,sigma=1):

	"""
	Apply (square) guassan kernel blur to image 

	"""
	size = int(kernel_size) // 2
	x, y = np.mgrid[-size:size+1, -size:size+1]
	normal = 1 / (2.0 * np.pi * sigma**2)
	kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

	return cv.filter2D(img, -1, kernel)

def cut(img,thresh=90):
	copy= img.copy()
	copy[copy > thresh] = 255
	copy[copy < thresh] = 0
	return copy


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


# imp.plot_grey(otsu(g_filter))

def remove_background_noise(img, dilations=1, kernel_size=5, thresh=174):
    kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    clean_image = cv.dilate(img, kern, iterations=dilations)
    # Remove all pixels that are brighter than thresh
    return clean_image * (clean_image < thresh)   

from sklearn.cluster import KMeans
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






#  
# imp.im_subplot([gray,otsu(gray),cut(gray),clean],shape=[1,4])
# clean = remove_background_noise(gray)
# g_filter=gaussian_filter(gray,8)


# imp.im_subplot([otsu(g_filter),otsu(clean)],shape=[1,3])









