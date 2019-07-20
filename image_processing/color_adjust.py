# -- color_adjust --

# 	rgb2bgr(RGB)

# 	bgr2rgb(BGR)

# 	rgb2gray(rgb)

# 	color_isolation(self,plot=False)

# 	intensity_plot(self)

# 	grey_level_adjust(self,grey_levels)

import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def rgb2bgr(RGB):

	R,G,B=RGB[:,:,0],RGB[:,:,1],RGB[:,:,2]
	return np.stack([B,G,R], axis=2)


def bgr2rgb(BGR):
	B,G,R= BGR[:,:,0],BGR[:,:,1],BGR[:,:,2]
	return np.stack([R,G,B], axis=2)

def rgb2yiq(RGB,norm=True):
	if norm:
		RGB=RGB/255.0
	R,G,B=RGB[:,:,0],RGB[:,:,1],RGB[:,:,2]
	y = (0.299*R + 0.587*G + 0.114*B)
	i = (0.59590059*R -0.27455667*G -0.32134392*B)
	q = (0.21153661*R -0.52273617*G + 0.31119955*B)
	return np.stack([y,i,q], axis=2)


def color_isolation(img):  
	"""
	Takes in an image and isolates it into 
	R,G,B color counter parts. 

	"""

	dim = np.zeros(img.shape[0:2]).astype(int)
	if len(img.shape)==2:
	    R,G,B= img,img,img
	    warnings.warn("Image should have 3 dims including color channels")
	else:
	    R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]

	# if plot ==True:
	# 	im_subplot ([np.stack((R,dim,dim), axis=2),np.stack((dim,G,dim), axis=2),
	#             np.stack((dim,dim,B), axis=2)],shape=[1,3], 
	#            titles=['R','G','B'] )
	
	return R,G,B

def mean_subtraction(img,sig=1):

	R,G,B= color_isolation(img)

	return np.stack([(R-R.mean())/sig,(G-G.mean())/sig,(B-B.mean())/sig], axis=2)


def intensity_plot(img):
	
    if len(img.shape) > 2:
        raise ValueError("Image must be 2D grey scale")
    # create the x and y coordinate arrays (here we just use pixel indices)
    
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # create the figure
    fig=plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1, 2, 1)
    plt.subplot(121)
    ax.set_title('Resized Image')
    plt.imshow(img)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('Intensity Plot')
    ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, 
            linewidth=0)
    ax.grid(False)
    ax.set_zticks([])
    ax.view_init(85, 0)
    plt.show()

def grey_level_adjust(img,grey_levels):
	"""
	color_range= 2^(#bits)
	Adjust grey scale color 
	"""
	grey_levels-=255
	grey_levels=abs(grey_levels)

	return (img/grey_levels).astype(int)*grey_levels









