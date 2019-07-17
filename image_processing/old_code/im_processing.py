import numpy as np
from scipy import fftpack
import cv2
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D




def cv_read(file_path):
	"""
	Process and Image RGB not BGR 
	"""
	image= cv2.imread(str(file_path))
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def plot_grey(im):

    plt.imshow(im,cmap='gray')
    plt.show()

def plot_hist(gray):

	if len(gray.shape)>2:
		raise ValueError('Must be Grey Scaled Image')
	plt.hist(gray.ravel(), bins=256, fc='k', ec='k')
	plt.show()

def hist_density(gray,thresh=128):
	his = np.histogram(gray, np.arange(0,257))[0]
	return np.sum(his[:thresh])/np.sum(his),np.sum(his[thresh:])/np.sum(his)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def im_subplot(ims,shape,titles=None,cmap= 'gray'):

	if titles == None:
	    titles =[str(" ") for i in range(len(ims))]
	for i,im in enumerate(ims): 
	    i+=1
	    ct= shape[0]*100 + shape[1]*10 + i
	    plt.subplot(ct)
	    plt.title(titles[i-1])
	    plt.imshow(im,cmap =cmap)
	plt.show()


class image:

	def __init__(self,im):
		self.img= im

	def rgb2gray(self):
		rgb=self.img
		r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
		gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
		return gray

	def zero_pad(pad_len=2):
		"""
		Zero Pad and IMage 
		"""
		return np.pad(self.img, (pad_len, pad_len), 'constant')


	def color_isolation(self,plot=False):  
		"""
		Takes in an image and isolates it into 
		R,G,B color counter parts. 

		"""

		dim = np.zeros(self.img.shape[0:2]).astype(int)
		if len(self.img.shape)==2:
		    R,G,B= self.img,self.img,self.img
		    warnings.warn("Image should have 3 dims including color channels")
		else:
		    R,G,B=self.img[:,:,0],self.img[:,:,1],self.img[:,:,2]
		    
		plt.figure(figsize=(12,10))



		if plot ==True:
			im_subplot ([np.stack((R,dim,dim), axis=2),np.stack((dim,G,dim), axis=2),
		            np.stack((dim,dim,B), axis=2)],shape=[1,3], 
		           titles=['R','G','B'] )

		else:
			return R,G,B

	def intensity_plot(self):
    	
	    if len(self.img.shape) > 2:
	        raise ValueError("Image must be 2D grey scale")
	    # create the x and y coordinate arrays (here we just use pixel indices)
	    
	    xx, yy = np.mgrid[0:self.img.shape[0], 0:self.img.shape[1]]

	    # create the figure
	    fig=plt.figure(figsize=(15,10))
	    ax = fig.add_subplot(1, 2, 1)
	    plt.subplot(121)
	    ax.set_title('Resized Image')
	    plt.imshow(self.img)
	    ax = fig.add_subplot(1, 2, 2, projection='3d')
	    ax.set_title('Intensity Plot')
	    ax.plot_surface(xx, yy, self.img ,rstride=1, cstride=1, 
	            linewidth=0)
	    ax.grid(False)
	    ax.set_zticks([])
	    ax.view_init(85, 0)
	    plt.show()
		    

	def zoom_dup(self,factor=2):

		"""
		Simple zoom by duplicating the number of pixels
		"""
		x= self.img.copy()
		x= np.repeat(x, factor, axis=1)
		x= np.repeat(x, factor, axis=0)
		return x

	def grey_level_adjust(self,grey_levels):
		"""
		color_range= 2^(#bits)
		Adjust grey scale color 
		"""
		grey_levels-=255
		grey_levels=abs(grey_levels)

		return (self.img/grey_levels).astype(int)*grey_levels

class sharpen:
	def __init__(self,img):
		self.img = img.astype(np.uint8)

	def sharpen(self):


		#generating the kernels
		kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

		#process and output the image
		return cv2.filter2D(self.img, -1, kernel)

	def excessive(self):

		#generating the kernels
		kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])

		#process and output the image
		return cv2.filter2D(self.img, -1, kernel)

	def edge_enhance(self):

		#generating the kernels
		kernel = np.array([[-1,-1,-1,-1,-1],
		                           [-1,2,2,2,-1],
		                           [-1,2,8,2,-1],
		                           [-2,2,2,2,-1],
		                           [-1,-1,-1,-1,-1]])/8.0

		#process and output the image
		return cv2.filter2D(self.img, -1, kernel)



class simple_kernel:

	def __init__(self,img):
		self.img = img.astype(np.uint8)

	def sharpen_filter(self):
		kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
		return cv2.filter2D(self.img, -1, kernel)

	def edge_filter(self):
		"""
		Simple kernel for edge detection 
		"""
		kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]).astype(float)
		return cv2.filter2D(self.img, -1, kernel)

	def gaussian_filter(self,kernel_size= 3 ,sigma=1):

		"""
		Apply (square) guassan kernel blur to image 

		"""

		size = int(kernel_size) // 2
		x, y = np.mgrid[-size:size+1, -size:size+1]
		normal = 1 / (2.0 * np.pi * sigma**2)
		kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

		return cv2.filter2D(self.img, -1, kernel)



class edge_finder:

	"""

	Canny image edge detection algo:

	4 parts:

	Guassoan Filter (Noise Reductoin) 

	Sobel Filter (Gradient Computation)
	Non-maximum Suppression 
	Double Threshold 
	Edge Tracknig by hystersis 



	"""


	def __init__(self,im,kernel_size=5,weak=25,strong=255):

		if len(np.shape(im))!=2:

			raise ValueError('**Must be 2D Grey Scale numpy.ndarray()\n')
		self.img= im.astype(float)


		self.kernel_size= kernel_size
		self.weak= np.int32(weak)
		self.strong = np.int32(strong)



	def gaussian_filter(self,sigma=1):

		"""
		Apply (square) guassan kernel blur to image 


		"""

		size = int(self.kernel_size) // 2
		x, y = np.mgrid[-size:size+1, -size:size+1]
		normal = 1 / (2.0 * np.pi * sigma**2)
		kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

		return cv2.filter2D(self.img, -1, kernel)

	def sobel_filters(self,img=None):

		"""
		Compute Intensity Gradient:: change in pixel intensity 

		Applies filters that highlight intensity change along x and y axis 

		Computes Gradient and Angle (Radians )
		"""

		if img is None:
			img = self.img

		Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
		Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

		Ix = cv2.filter2D(img, -1, Kx)
		#convolve x kernel 
		Iy = cv2.filter2D(img, -1, Ky)
		#convolve y kernel 

		G = np.hypot(Ix, Iy) #SQRT of SUM of SQUARES
		G = G / G.max() * 255 #norm to 8 bit
		theta = np.arctan2(Iy, Ix)
    
		return (G, theta) #G is gradient matrix, and theta angle 

	def non_max_suppression(self,gradients=None, theta_matrix=None):
		"""

		Edge thinning algo used when gradient filter is computed 

		For every pixel gradient (g) and angle (a):
			edge direction = line through pixel at angle a
			if pixel on edge > current:
				set pixel = 255 (white)
				other pixels= 0 (black)

		"""

		if gradients is None and theta_matrix is None:
			gradients, theta_matrix = self.sobel_filters()


		M, N = gradients.shape
		Z = np.zeros((M,N), dtype=np.int32)# create zero matix 
		angle = theta_matrix * 180. / np.pi #convert to deg
		angle[angle < 0] += 180 #add 180 wherever angle (-)


		for i in range(1,M-1): #for all rows
			for j in range(1,N-1): #for all columns 
				try:
					q = 255
					r = 255

				#angle 0
					if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
						q = gradients[i, j+1]
						r = gradients[i, j-1]
					#angle 45

					elif (22.5 <= angle[i,j] < 67.5):
						q = gradients[i, j+1]
						r = gradients[i, j-1]

					elif (67.5 <= angle[i,j] < 112.5):
						q = gradients[i+1, j]
						r = gradients[i-1, j]

					elif (112.5 <= angle[i,j] < 157.5):
						q = gradients[i-1, j-1]
						r = gradients[i+1, j+1]

					if (gradients[i,j] >= q) and (gradients[i,j] >= r):
						Z[i,j] = gradients[i,j]
					else:
						Z[i,j] = 0
				except IndexError as e:
				    pass
		return Z


	def threshold(self, img=None, lowThresholdRatio=0.05, highThresholdRatio=0.09):
		"""
		Set max pixel and min pixel intensity:

		Strong pixel = p > max threshold  || ~ 10%+ of brightest pixel 
		Weak pixel = max threshold  > p > min threshold  || ~ .05 * .09%  of brightest pixel 
		Irrelvant Pixel=  min threshold  > p

		Sets pixel vals to 0,min,and max vals 

		"""
		if img is None:
			img = self.img


		highThreshold = img.max() * highThresholdRatio;
		lowThreshold = highThreshold * lowThresholdRatio;

		M, N = img.shape
		res = np.zeros((M,N), dtype=np.int32) ##irrelevant is - 

		strong_i, strong_j = np.where(img >= highThreshold)
		zeros_i, zeros_j = np.where(img < lowThreshold)

		weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

		res[strong_i, strong_j] = self.strong
		res[weak_i, weak_j] = self.weak

		return res

	def hysteresis(self,img=None):
		"""
		If strong pixel neighbors weak, convert weak to strong 

		"""

		if img is None:
			img = self.img

		M, N = img.shape  
		for i in range(1, M-1):
		    for j in range(1, N-1):
		        if (img[i,j] == self.weak):
		            try:
		                if ((img[i+1, j-1] == self.strong) or (img[i+1, j] == self.strong) or (img[i+1, j+1] == self.strong)
		                    or (img[i, j-1] == self.strong) or (img[i, j+1] == self.strong)
		                    or (img[i-1, j-1] == self.strong) or (img[i-1, j] == self.strong) or (img[i-1, j+1] == self.strong)):
		                    img[i, j] = self.strong
		                else:
		                    img[i, j] = 0
		            except IndexError as e:
	                	pass

		return img


	def isolate(self):

		self.smoothed= self.gaussian_filter()
		self.gradients, self.theta_matrix = self.sobel_filters(self.smoothed)
		self.suppressed = self.non_max_suppression(self.gradients, self.theta_matrix)
		self.threshold= self.threshold(self.suppressed)
		return self.hysteresis(self.threshold)


def convolve_2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    m, n = kernel.shape
    if (m == n):
        y, x = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((y,x))
        for i in range(y):
            for j in range(x):
                new_image[i][j] = np.sum(image[i:i+m, j:j+m]*kernel) 
    return new_image


def crop(img, x_left=0,x_right=0,y_bot=0,y_up=0):
	"""
	Crop function 
	"""
	return img[y_up:img.shape[0]-y_bot, x_left:img.shape[1]-x_right]	

def zero_pad(img,pad_len=2):
	"""
	Zero Pad and IMage 
	"""
	return np.pad(img, (pad_len, pad_len), 'constant')
 








		
