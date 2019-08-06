
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
from .color_adjust import rgb2gray,bgr2rgb,rgb2bgr,rgb2yiq,hist_density
import warnings 

def cv_read(file_path,color_scale='RGB'):
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

	if color_scale not in ['RGB','BGR','YIQ','GRAY']:
		raise ValueError("Color Scale must be one of followng:: 'RGB','BGR','YIQ','GRAY'")
	elif color_scale=='RGB': return bgr2rgb(image)
	elif color_scale=='BGR': return image
	elif color_scale=='YIQ': return rgb2yiq(bgr2rgb(image))
	elif color_scale=='GRAY': return rgb2gray(bgr2rgb(image))



def plot_grey(im,title=None,xlabel=None,ylabel=None,to_RGB=False):
	"""
	Simple matplotlib image plotter. 
	Takes in simple matplotlib args 
	and plots grays and colored images 
	easier without having to specify. 

	im:: RGB image numpy array only
	returns:img 
	"""

	if to_RGB: im= bgr2rgb(im)
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

def __get_grid_shape(l,max_range):

	h= int(len(l)/max_range) + 1

	return [h,max_range]

def im_subplot(ims,shape=None,titles=None,cmap= 'gray',suptitle=None,plot=True,to_RGB=False):
	"""
	Basic Subplotting Function. 
	
	input: list of images 

	returns subplot of images plotting next to each other

	"""


	if shape == None:
		if len(ims)<4:
			shape = [1,len(ims)]
		else:
			shape= __get_grid_shape(ims,max_range=4)

	if titles == None:
	    titles =[str(" ") for i in range(len(ims))]

	if len(ims)!=len(titles):
		raise ValueError('Number of Images must equal Number of Titles')
	fig = plt.figure(1)
	if suptitle: plt.suptitle(suptitle)
	for i in range(1,len(ims)+1): 
	    fig.add_subplot(shape[0],shape[1],i)
	    plt.title(titles[i-1])
	    if to_RGB: 
	    	image= bgr2rgb(ims[i-1])
	    else: 
	    	image = ims[i-1]
	    plt.imshow(image,cmap =cmap)
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

class im_stats:
	'''
	Basic Image Stats Class 
	'''

	def __init__(self,img):
		self.img,self.flat= img,img.flatten()
		self.shape = img.shape
		self.min,self.max=self.img.min(),self.img.max()
			

	def describe_all(self,per_channel=True):
		"""
		Show all stats options.
		per_channels gives per channels stats if 
		true and image is colored or has channels 


		"""

		if len(self.shape)==3:
			ctype= '(Colored Image)'
			self.color= True 
		elif len(self.shape)==2:
			ctype= '(Gray Image)'
			self.color= None

		print('########## Image Stats ##########')
		print('Image Size',self.shape,'--->',ctype)


		if per_channel and self.color:
			channels = [self.img[:,:,0],self.img[:,:,1],self.img[:,:,2]]
			for i,c in enumerate(channels):
				i+=1
				self.img=c
				print('\n','-+---------Channel-------+',i,'\n')

				self.general()
				self.distrib_stat()
				self.percentile()
		else:
			self.general()
			self.distrib_stat()
			self.percentile()	

	def general(self):
		'''
		General Stats including 
		min,max, and root mean square 

		'''
	
		print('Min:',self.min)
		print('Max:',self.max)
		self.rms=np.sqrt(np.sum(np.square(self.flat))/len(self.flat))
		print('RMS: {:.2f}'.format(self.rms))


	def distrib_stat(self):
		"""
		Get Simple General Distribution stats 

		"""
		print('----------------------')
		print('Distribution:')

		half=(self.max)/2
		print('Intensity: {:.2f} +/- {:.2f}'.format(self.img.mean(),self.img.std()))
		vals = hist_density(self.img,half)
		print('{:.2f}% of pixels > {}'.format(vals[1]*100,half))

	def percentile(self):
		"""
		Get Quartile Boxplot 
		pixel intensity stats 

		"""
		print('----------------------')
		print('Percentiles:')
		quarter,half,quarter_3,max_val= (self.max)/4,(self.max)/2,(3*self.max)/4,(self.max)
		name,count = ['25% |','50% |','75% |','100%|'],[quarter,half,quarter_3,max_val]
		previous =0 
		prev_perc= 0 
		for n,c in enumerate(count):
			vals = hist_density(self.img,c)
			prev_perc = vals[0]-prev_perc
			print(name[n],'{:.2f}% of pixels (~ {} < i < {})'.format((prev_perc)*100,previous,c))
			previous=c



			
















