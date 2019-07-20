# -- im_processing -- + Fourier Analysis Class 

# 	cv_read(file_path)

# 	plot_grey(im)

# 	plot_hist(gray)

# 	im_subplot(ims,shape,titles=None,cmap= 'gray')

# 	zero_pad(pad_len=2)

# 	hist_density(gray,thresh=128)

# 	zoom_dup(self,factor=2)

# 	crop(img, x_left=0,x_right=0,y_bot=0,y_up=0)

# 	cut(img,thresh=90)

# 	resize(image, width = None, height = None, inter = cv2.INTER_AREA):

import numpy as np
import cv2
import matplotlib.pyplot as plt 
import sys

# import any special Python 3 packages
if sys.version_info.major == 3:
    from urllib.request import urlopen



def cv_read(file_path):
	"""
	Process and Image RGB not BGR 
	"""
	image= cv2.imread(str(file_path))
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def plot_grey(im):

    plt.imshow(im,cmap='gray')
    plt.show()

def cv_plot(img,title= ' '):
	cv2.imshow(title,img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def plot_hist(gray):

	if len(gray.shape)>2:
		raise ValueError('Must be Grey Scaled Image')
	plt.hist(gray.ravel(), bins=256, fc='k', ec='k')
	plt.show()

def hist_density(gray,thresh=128):
	his = np.histogram(gray, np.arange(0,257))[0]
	return np.sum(his[:thresh])/np.sum(his),np.sum(his[thresh:])/np.sum(his)

def im_subplot(ims,shape=None,titles=None,cmap= 'gray'):
	if shape == None:
		shape = [1, len(ims)]
	if titles == None:
	    titles =[str(" ") for i in range(len(ims))]


	fig = plt.figure(1)
	for i in range(1,len(ims)+1): 
	    fig.add_subplot(shape[0],shape[1],i)
	    plt.title(titles[i-1])
	    plt.imshow(ims[i-1],cmap =cmap)
	plt.show()

# def im_subplot(ims,shape=None,titles=None,cmap= 'gray'):
# 	if shape == None:
# 		shape = [1, len(ims)]
# 	if titles == None:
# 	    titles =[str(" ") for i in range(len(ims))]
# 	for i in range(shape[0]):
# 		for j in range(shape[1]):
# 			print(i,j)
# 			# plt.subplot2grid((shape[0],shape[1]), (i,j))
# 			# plt.title(titles[i-1])
# 			# plt.imshow(ims[i],cmap =cmap)
# 	plt.show()

def zoom_dup(img,factor=2):

	"""
	Simple zoom by duplicating the number of pixels
	"""
	x= img.copy()
	x= np.repeat(x, factor, axis=1)
	x= np.repeat(x, factor, axis=0)
	return x


def cut(img,thresh=90):
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
	Zero Pad and IMage 
	"""
	return np.pad(img, (pad_len, pad_len), 'constant')

def __pad(vector, pad_width, iaxis, kwargs):
	pad_value = kwargs.get('padder', 10)
	vector[:pad_width[0]] = pad_value
	vector[-pad_width[1]:] = pad_value
	return vector

def pad_with(img, pad_len=2, val=10):

	return np.pad(img, pad_len, __pad, padder=val)

def crop(img, x_left=0,x_right=0,y_bot=0,y_up=0):
	"""
	Crop function 
	"""
	return img[y_up:img.shape[0]-y_bot, x_left:img.shape[1]-x_right]	



#tues 11-11:30 
# shape= 3,4
# for i in range(1,15): 
# 	ct= shape[0]*100 + shape[1]*10 + i
# 	if ct % 10 !=0: 
# 		print(ct)
# 		continue 
# 	print('e')


