
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
	    return ("Image should have 3 dims including color channels")
	else:
	    R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
	    return R,G,B

def mean_subtraction(img,sig=1):

	R,G,B= color_isolation(img)

	return np.stack([(R-R.mean())/sig,(G-G.mean())/sig,(B-B.mean())/sig], axis=2)

class intensity_plot:

    def __init__(self,img,resize=.5,cmap='plasma'):
        """
        Create Intensity Plot Class 

        """
        if len(img.shape) > 2:
            img = rgb2gray(img)
            print("Image must be 2D grey scale... Gray Scaling...")
        # create the x and y coordinate arrays (here we just use pixel indices)
        self.img= img 
        self.resize = resize
        self.cmap=cmap
        if self.resize and sum(img.shape)>250:
            self.img = self.__resize()
            print('Resizing {}....... New Shape'.format(img.shape),self.img.shape)
        elif sum(img.shape)<250:
        	print('Image too small to Resize ') 
        self.xx, self.yy = np.mgrid[0:self.img.shape[0], 0:self.img.shape[1]]
        
        
    def __resize(self, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and grab the image size
        dim = tuple([int(self.resize * i) for i in self.img.shape[0:2]])
        # resize the image
        return cv2.resize(self.img, (dim[1],dim[0]), interpolation = inter)

    def show(self,view=(85, 0),title='Pixel Intensity',figsize=(15,10),plot=True):
        # create the figure
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(str(title))
        surf= ax.plot_surface(self.xx, self.yy, self.img ,rstride=1, cstride=1, cmap=self.cmap, linewidth=0, antialiased=False)
        ax.view_init(view[0], view[1])
        ax.grid(False)

        cbar= fig.colorbar(surf, shrink=0.5, aspect=5)
        cbar.set_label("Pixel Intensity")
        ax.set_zticks([])
        if plot: plt.show()
        
def plot_hist(img,hist_density_thresh=None,show=True):
	"""
	Generates an image histogram pixel intensities. An shows ratio

	- Input:
	     - im:: gray scaled image (np.array). Colored images are automatically gray scaled 
	     - hist_density_thresh:: None. Else int in range [0, 255] that shows desired threshold
	     - show:: show plot 

	- Output:
	     - histogram plot 
     """

	if len(img.shape)>2:
		print('Colored Img')
		R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
		if hist_density_thresh:
			vals_R= hist_density(R,thresh=hist_density_thresh)
			title_R = '{:.2f}% of R pixels are brighter {}'.format(vals_R[1]*100,hist_density_thresh)
			vals_G= hist_density(B,thresh=hist_density_thresh)
			title_G = '{:.2f}% of G pixels are brighter {}'.format(vals_G[1]*100,hist_density_thresh)
			vals_B= hist_density(G,thresh=hist_density_thresh)
			title_B = '{:.2f}% of B pixels are brighter {}'.format(vals_B[1]*100,hist_density_thresh)
			
			plt.subplot(131)
			plt.title(title_R)
			plt.hist(R.ravel(), bins=256, color = "red")
			plt.axvline(hist_density_thresh)
			plt.ylabel('Freq of Pixel Intensity')
			plt.xlabel('Pixel Initensity [0,255]')

			plt.subplot(132)
			plt.title(title_B)
			plt.hist(B.ravel(), bins=256, color = "blue")
			plt.axvline(hist_density_thresh)
			plt.ylabel('Freq of Pixel Intensity')
			plt.xlabel('Pixel Intensity [0,255]')

			plt.subplot(133)
			plt.title(title_B)
			plt.hist(G.ravel(), bins=256,color = "green")
			plt.axvline(hist_density_thresh)
			plt.ylabel('Freq of Pixel Intensity')
			plt.xlabel('Pixel Intensity [0,255]')
			

		else: 
			plt.title('Color Histogram')
			plt.hist(R.ravel(), bins=256, color = "red")
			plt.hist(B.ravel(), bins=256, color = "blue")
			plt.hist(G.ravel(), bins=256, color = "green")
			plt.ylabel('Freq of Pixel Intensity')
			plt.xlabel('Pixel Intensity [0,255]')

		if show: plt.show()

	else:
		print('Img Assummed to be Gray')
		if hist_density_thresh:
			vals= hist_density(img,thresh=hist_density_thresh)
			title = '{:.2f}% of pixels are brighter {}'.format(vals[1]*100,hist_density_thresh)
			plt.title(title)

		else: plt.title('Gray Histogram')
		plt.hist(img.ravel(), bins=256, fc='k', ec='k')
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

def grey_level_adjust(img,grey_levels,plot=True):
	"""
	color_range= 2^(#bits)
	Adjust grey scale color 
	"""
	grey_levels-=255
	grey_levels=abs(grey_levels)

	return (img/grey_levels).astype(int)*grey_levels

def brighten(img,alpha,beta):
	"""
	Simple brightness function based on 
	brighted = alpha*img+beta
	"""
	return np.clip(alpha*img + beta, 0, 255).astype(int)








