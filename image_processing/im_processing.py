
############################################
# Im Processing::                          #
# Simple numpy based functions for image   #
# restructuring and editing. It functions  #
# for reading images, plotting,cropping,   #
# thresholding, simple histogram analysis  #
# zooming and resizing and padding. Basic  #
# initial plotting and editing tools       #
############################################

import numpy as np
import cv2
import matplotlib.pyplot as plt 
from .color_adjust import rgb2gray,bgr2rgb,rgb2bgr
import warnings 

def cv_read(file_path,RGB=True):
	"""
	Read jpg,jpeg,png image files. 
	Can be read as RGB or default 
	openCV read as BGR img. 

	file_path:: path to img file 
	RGB:: True --> return RGB, else BGR
	"""

	image= cv2.imread(str(file_path))
	if image is None:
		raise ValueError('No image found at {}'.format(file_path))
	if RGB: return bgr2rgb(image)
	else: return img 


def plot_grey(im,title=None,xlabel=None,ylabel=None,convert_RGB=False):
	"""
	Simple matplotlib image plotter. 
	Takes in simple matplotlib args 
	and plots grays and colored images 
	easier without having to specify. 

	im:: RGB image numpy array only
	returns:img 
	"""

	if convert_RGB: img= bgr2rgb(img)
	if title: 
		plt.title(str(title))
	elif xlabel: 
		plt.xlabel(str(xlabel))
	elif ylabel: 
		plt.ylabel(str(ylabel))

	plt.imshow(im,cmap='gray')
	plt.show()

def cv_plot(img,title= ' ',convert_BGR=False):
	"""
	Simple cv based image plotter. 
	Plots grays and colored images 
	easier without having to specify. 

	im:: BGR image
	returns:img 
	"""
	if convert_BGR: img= rgb2bgr(img)
	cv2.imshow(title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def plot_hist(gray,hist_density_thresh=None,show=True):

	if len(gray.shape)>2:
		warnings.warn('Gray Scaling Image...')
		gray= rgb2gray(gray)

	if hist_density_thresh:
		vals= hist_density(gray,thresh=hist_density_thresh)
		title = '{:.2f}% of pixels are brighter {}'.format(vals[1]*100,hist_density_thresh)
		plt.title(title)

	else: plt.title('Color Histogram')
	if len(gray.shape)>2:
		raise ValueError('Must be Grey Scaled Image')
	plt.hist(gray.ravel(), bins=256, fc='k', ec='k')
	if hist_density_thresh: plt.axvline(hist_density_thresh)
	if show: plt.show()

def hist_density(gray,thresh=128):
	"""
	Illustrates the percent of images above
	and below a set threshold. 

	input:: gray scaled image
	thresh:: threshold 

	return (% below thresh, % above thresh)
	"""
	his = np.histogram(gray, np.arange(0,257))[0]
	return np.sum(his[:thresh])/np.sum(his),np.sum(his[thresh:])/np.sum(his)

def im_subplot(ims,shape=None,titles=None,cmap= 'gray',suptitle=None,plot=True):
	"""
	Basic Subplotting Function. 
	
	input: list of images 

	returns subplot of images plotting next to each other

	"""

	if shape == None:
		shape = [1, len(ims)]
	if titles == None:
	    titles =[str(" ") for i in range(len(ims))]

	if len(ims)!=len(titles):
		raise ValueError('Number of Images must equal Number of Titles')
	fig = plt.figure(1)
	if suptitle: plt.suptitle(suptitle)
	for i in range(1,len(ims)+1): 
	    fig.add_subplot(shape[0],shape[1],i)
	    plt.title(titles[i-1])
	    plt.imshow(ims[i-1],cmap =cmap)
	if plot: plt.show()


def zoom_dup(img,factor=2):

	"""
	Simple zoom by duplicating the number of pixels
	"""
	x= img.copy()
	x= np.repeat(x, factor, axis=1)
	x= np.repeat(x, factor, axis=0)
	return x


def cut(img,thresh=90):
	"""
	Set images above thresh to white 
	and below to black 
	"""
	copy= img.copy()
	copy[copy > thresh] = 255
	copy[copy < thresh] = 0
	return copy

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image

    if width is None and height is None:
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
    return cv2.resize(image, dim, interpolation = inter)


def zero_pad(img,pad_len=2):
	"""
	Zero Pad image 
	"""
	return np.pad(img, (pad_len, pad_len), 'constant')

def __pad(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value
	return vector

def pad_with(img, pad_len=2, val=10):

	"""
	pad image with set pad length and value 

	"""
	if pad_len==0:
		return img 

	if len(img.shape)>2:
		dims = [img[:,:,0],img[:,:,1],img[:,:,2]]
		final_padded = [] 
		for dim in dims:
			final_padded.append(np.pad(dim, pad_len, __pad, padder=val))
		print(len(final_padded))
		return np.stack(final_padded, axis=2)
	else: 
		return np.pad(img, pad_len, __pad, padder=val)

def crop(img, x_left=0,x_right=0,y_bot=0,y_up=0):
	"""
	Crop function 
	"""
	return img[y_up:img.shape[0]-y_bot, x_left:img.shape[1]-x_right]	

def rot45(array):
	"""
	Crude Image 45 deg Rotation Function 
	"""
	rot = []
	for i in range(len(array)):
	    rot.append([1] * (len(array)+len(array[0])-1))
	    for j in range(len(array[i])):
	        #rot[i][int(i + j)] = array[i][j]
	        rot[i][int(i + j)] = array[i][j]

	return np.array(rot)

