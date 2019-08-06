############################################
# Fourier Analysis::`					   #
# FA is effectively the Decomposition      #
# of nd signal (Image) into sinusoid       #
# functions. The primay objective is       #
# to translate a signal from the time      #
# domain to the frequency one. This will   #
# reveal the periodic nature of the image. # 
############################################

import numpy as np
from numpy import arctan2, fliplr, flipud
import matplotlib.pyplot as plt
from .color_adjust import rgb2gray,rgb2bgr
from .kernel import gaussian_filter
import cv2

def magnitude_spectrum(img,plot=False):
	"""
	Generated FFT Plot of image 
	"""

	if len(img.shape)>2:
		img= rgb2gray(img)
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = np.log(np.abs(fshift))
	if plot:
		plt.imshow(magnitude_spectrum, cmap = 'gray')
		plt.show()
	return magnitude_spectrum


def gen_square_im(size=(20,20),n=2,val=1):
	"""
	Generates a gray scaled square image in white background
	Input::
		size:: image size 
		n:: distance from the sides 
		val:: what pixel val color of square 
	"""
	inp = np.zeros(size)
	inp[n,n:inp.shape[1]-n] = val*np.ones(inp.shape[1]-2*n) #top bar 
	inp[inp.shape[0]-(n+1),n:inp.shape[1]-n] = val*np.ones(inp.shape[1]-2*n) #bottom bar 
	inp[n:inp.shape[0]-n,n] = val*np.ones(inp.shape[0]-2*n)
	inp[n:inp.shape[0]-n,inp.shape[1]-(n+1)] = val*np.ones(inp.shape[0]-2*n)
	return inp

def gen_line_im(size=(100,100),samp_freq=5,vert=True,checker=False,balance=False):
	"""
	Generates a gray scaled lined image 
	Input::
		size:: image size 
		samp_freq:: frequency lines per iteration (lower values == more lines)
		vert: True --> Vert Lines False --> Horizontal 
		checker: True --> overlap the lines 
		balance: creates an even image 
	"""

	if size[0] % samp_freq !=0 or size[1] % samp_freq !=0:
		raise ValueError("FREQ MUST BE MULTIPLE OF SIZE")

	inp = 125*np.ones(size)
	for n in range(0,size[1],2*samp_freq):
		inp[n:n+samp_freq] = 255*np.ones((samp_freq,inp.shape[1])) #top bar 

	if checker: return inp+np.rot90(inp)
	if balance: 
		bal =np.concatenate([inp,np.ones((samp_freq,size[1]))],axis=0) 
	else: bal=inp
	if vert: return np.rot90(bal)
	else: return bal

class im_noise:
	"""
	Class of various forms of image noise
	"""
	def __init__(self,img):
		self.img= img
		if len(self.img.shape)==2:
			self.color = False 
		elif len(self.img.shape)==3:
			self.color = True
		else:
			raise SizeError('Invalid Image Shape')

	def gaussian(self,mean=0,sigma=.1):
		"""
		Adding gausian distribution of noise
		"""
		if self.color:
			row,col,ch= self.img.shape
			gauss = np.random.normal(mean,sigma,(row,col,ch))
			return self.img + gauss
		else:
			row,col= self.img.shape
			gauss = np.random.normal(mean,sigma,(row,col))
			return self.img + gauss

	def salt_pepper(self,s_vs_p = 0.5, amount = 0.004):
		"""
		Add salt_pepper noise aka Replaces random pixels with 0 or 1.
		s_vs_p:: ratio of salt/pepper
		amount:: amount of noise
		"""
		copy = self.img.copy()
		#Salt
		num_salt = np.ceil(amount * self.img.size * s_vs_p)
		coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
		copy[tuple(coords[0].tolist()),tuple(coords[1].tolist())] = 1

		# Pepper 
		num_pepper = np.ceil(amount* self.img.size * (1. - s_vs_p))
		coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
		copy[tuple(coords[0].tolist()),tuple(coords[1].tolist())] = 0
		return copy

	def poisson(self):
		"""
		Adding poisson distribution of noise
		"""
		return self.img + np.random.poisson(self.img)
		
	def speckle(self):
		"""
		Speck distribution of noise
		Multiplicative noise using out = image + n*image,where
        n is uniform noise with specified mean & variance.
		"""
		if self.color:
			row,col,ch= self.img.shape
			gauss = np.random.randn(row,col,ch)
			gauss = gauss.reshape(row,col,ch)        
			return self.img + self.img * gauss
		else:
			row,col= self.img.shape
			gauss = np.random.randn(row,col)
			gauss = gauss.reshape(row,col) 
			return self.img + self.img * gauss
class polar:

	def __init__(self,img,resize= .1,plot=True,canny_min=100,canny_max=200,to_BGR=False):
		
		self.og_img = img 

		if to_BGR:
			print('Converting to CV BGR Format....')
			img = rgb2bgr(img)

		if len(img.shape)>2:
			print('Colored Image... Gray Scaling....')
			img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		img = cv2.Canny(img,canny_min,canny_max) 
		self.edge_im= img
		print('Running Canny Edge Detection...')
		dim = tuple([int(resize * i) for i in img.shape[0:2]])
		print('Resizing to {}% of image...'.format(resize*100))
		self.img = cv2.resize(img, (dim[1],dim[0]), interpolation =cv2.INTER_AREA)

		self.diag_len = int((self.img.shape[0]**2+self.img.shape[1]**2)**(1/2))

	def hough_line(self,angle_step=1,value_threshold=5,plot=True):
	    """
	    Hough transform for lines
	    Input:
	    img - 2D binary image with nonzeros representing edges
	    angle_step - Spacing between angles to use every n-th angle
	                 between -90 and 90 degrees. Default step is 1.
	    lines_are_white - boolean indicating whether lines to be detected are white
	    value_threshold - Pixel values above or below the value_threshold are edges
	    Returns:
	    accumulator - 2D array of the hough transform accumulator
	    theta - array of angles used in computation, in radians.
	    rhos - array of rho values. Max size is 2 times the diagonal
	           distance of the input image.
	    """
	    # Rho and Theta ranges


	    thetas = (np.arange(-90.0, 90.0, angle_step))*np.pi/180

	    rhos = np.linspace(-self.diag_len, self.diag_len, self.diag_len * 2)

	    # Cache some resuable values
	    sin, cos = np.sin(thetas),np.cos(thetas)
	  
	    # Hough accumulator array of theta vs rho
	    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
	   
	    edges= self.img > value_threshold
	    edges_y,edges_x= np.nonzero(edges)

	    print('Translating Coordinated to Polar Space...')

	    # Vote in the hough accumulator
	    for i in range(len(edges_x)):

	        for t_idx in range(len(thetas)):
	            # Calculate rho. diag_len is added for a positive index
	            rho = self.diag_len + int(round(edges_x[i] * cos[t_idx] + edges_y[i] * sin[t_idx]))
	            accumulator[rho, t_idx] += 1

	    if plot:
	    	fig, ax = plt.subplots(1, 2, figsize=(10, 10))
	    	ax[0].imshow(self.edge_im)
	    	ax[0].set_title('Canny Grayscaled + Edged image')
	    	ax[0].axis('image')

	    	plt.imshow(
	    	accumulator, cmap='ocean_r',
	    	extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
	    	ax[1].set_aspect('equal', adjustable='box')
	    	ax[1].set_title('Hough transform')
	    	ax[1].set_xlabel('Angles (degrees)')
	    	ax[1].set_ylabel('Distance (pixels)')
	    	ax[1].axis('image')
	    	plt.show()
	    return accumulator, thetas, rhos




# class im_PCA:
#     """A method for doing dimensionality reduction by transforming the feature
#     space to a lower dimensionality, removing correlation between features and
#     maximizing the variance along each feature axis. This class is also used throughout
#     the project to plot data.
#     """
#     def __init__(self,img): 
#         self.img = img 

#     def cv_matrix(self): 
#         n_samples = np.shape(self.img)[0]
#         covariance_matrix = (1 / (n_samples-1)) * (self.img - self.img.mean(axis=0)).T.dot(self.img - self.img.mean(axis=0))
#         return covariance_matrix
#     def transform(self, n_components):
#         """ Fit the dataset to the number of principal components specified in the
#         constructor and return the transformed dataset """
#         covariance_matrix = self.cv_matrix()

#         # Where (eigenvector[:,0] corresponds to eigenvalue[0])
#         eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

#         # Sort the eigenvalues and corresponding eigenvectors from largest
#         # to smallest eigenvalue and select the first n_components
#         idx = eigenvalues.argsort()[::-1]
#         eigenvalues = eigenvalues[idx][:n_components]
#         eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]

#         # Project the data onto principal components
#         X_transformed = self.img.dot(eigenvectors)
        # return X_transformed



def gradient(image, same_size=False):
    """ Computes the Gradients of the image separated pixel difference
    
    Gradient of X is computed using the filter 
        [-1, 0, 1]
    Gradient of X is computed using the filter 
        [[1,
          0,
          -1]]
    Parameters
    ----------
    image: image of shape (imy, imx)
    same_size: boolean, optional, default is True
        If True, boundaries are duplicated so that the gradients
        has the same size as the original image.
        Otherwise, the gradients will have shape (imy-2, imx-2)
        
    Returns
    -------
    (Gradient X, Gradient Y), two numpy array with the same shape as image
        (if same_size=True)
    """
    sy, sx = image.shape
    if same_size:
        gx = np.zeros(image.shape)
        gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
        gx[:, 0] = -image[:, 0] + image[:, 1]
        gx[:, -1] = -image[:, -2] + image[:, -1]
    
        gy = np.zeros(image.shape)
        gy[1:-1, :] = image[:-2, :] - image[2:, :]
        gy[0, :] = image[0, :] - image[1, :]
        gy[-1, :] = image[-2, :] - image[-1, :]
    
    else:
        gx = np.zeros((sy-2, sx-2))
        gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

        gy = np.zeros((sy-2, sx-2))
        gy[:, :] = image[:-2, 1:-1] - image[2:, 1:-1]
    
    return gx, gy


def magnitude_orientation(gx, gy):
    """ Computes the magnitude and orientation matrices from the gradients gx gy
    Parameters
    ----------
    gx: gradient following the x axis of the image
    gy: gradient following the y axis of the image
    
    Returns 
    -------
    (magnitude, orientation)
    
    Warning
    -------
    The orientation is in degree, NOT radian!!
    """
        
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / np.pi) % 360
            
    return magnitude, orientation


def compute_coefs(csx, csy, dx, dy, n_cells_x, n_cells_y):
    """
    Computes the coefficients for the bilinear (spatial) interpolation
    
    Parameters
    ----------
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    
    Notes
    -----
    We consider an image: image[y, x] (NOT image[x, y]!!!)
    
    /!csx and csy must be even number
    
    Using the coefficients
    ----------------------
    The coefficient correspond to the interpolation in direction of the upper left corner of the image.
    In other words, if you interpolate img, and res is the result of your interpolation, you should do
    
    res = zeros(n_cells_y*pixels_per_cell, n_cells_x*pixels_per_cell)
        with (csx, csy) the number of pixels per cell
         and dx, dy = csx//2, csy//2
    res[:-dx, :-dy] += img[dx:, dy:]*coefs
    
    then you rotate the coefs and do the same thing for every part of the image
    """
    if csx != csy:
        raise NotImplementedError("For now compute_coefs is only implemented for squared cells (csx == csy)")
        
        ################################
        #####     /!\ TODO  /!|    #####
        ################################

    else: # Squared cells
        # Note: in this case, dx = dy, we differentiate them only to make the code clearer
        
        # We want a squared coefficients matrix so that it can be rotated to interpolate in every direction
        n_cells = max(n_cells_x, n_cells_y)
        
        # Every cell of this matrix corresponds to (x - x_1)/dx 
        x = (np.arange(dx)+0.5)/csx
        
        # Every cell of this matrix corresponds to (y - y_1)/dy
        y = (np.arange(dy)+0.5)/csy
        
        y = y[np.newaxis, :]
        x = x[:, np.newaxis]

        # CENTRAL COEFFICIENT
        ccoefs = np.zeros((csy, csx))

        ccoefs[:dy, :dx] = (1 - x)*(1 - y)
        ccoefs[:dy, -dx:] = fliplr(y)*(1 - x)
        ccoefs[-dy:, :dx] = (1 - y)*flipud(x)
        ccoefs[-dy:, -dx:] = fliplr(y)*flipud(x)

        coefs = np.zeros((csx*n_cells - dx, csy*n_cells - dy))
        coefs[:-dy, :-dx] = np.tile(ccoefs, (n_cells - 1, n_cells - 1))

        # REST OF THE BORDER
        coefs[:-dy, -dx:] = np.tile(np.concatenate(((1 - x), np.flipud(x))), (n_cells - 1, dy))
        coefs[-dy:, :-dx] = np.tile(np.concatenate(((1 - y), np.fliplr(y)), axis=1), (dx, n_cells - 1))
        coefs[-dy:, -dx:] = 1

        return coefs


def interpolate_orientation(orientation, sx, sy, nbins, signed_orientation):
    """ interpolates linearly the orientations to their corresponding bins
    Parameters
    ----------
    sx: int
        number of columns of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    nbins : int, optional, default is 9
        Number of orientation bins.
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    Returns
    -------
    pre-histogram: array of shape (sx, sy, nbins)
            contains the pre histogram of orientation built using linear interpolation
            to interpolate the orientations to their bins
    """
    
    if signed_orientation:
        max_angle = 360
    else:
        max_angle = 180
    
    b_step = max_angle/nbins
    b0 = (orientation % max_angle) // b_step
    b0[np.where(b0>=nbins)]=0
    b1 = b0 + 1
    b1[np.where(b1>=nbins)]=0
    b = np.abs(orientation % b_step) / b_step
    
    #linear interpolation between the bins
    # Coefficients corresponding to the bin interpolation
    # We go from an image to a higher dimension representation of size (sizex, sizey, nbins)
    temp_coefs = np.zeros((sy, sx, nbins))
    for i in range(nbins):
        temp_coefs[:, :, i] += np.where(b0==i, (1 - b), 0)
        temp_coefs[:, :, i] += np.where(b1==i, b, 0)
    
    return temp_coefs


def per_pixel_hog(image, dy=2, dx=2, signed_orientation=False, nbins=9, flatten=False, normalise=True):
    """ builds a histogram of orientation for a cell centered around each pixel of the image
    
    Parameters
    ---------
    image: numpy array of shape (sizey, sizex)
    dx   : the cell around each pixel in the x axis will have size 2*dx+1
    dy   : the cell around each pixel in the y axis will have size 2*dy+1
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.
        
    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)
    
    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    """
    gx, gy = gradient(image, same_size=True)
    magnitude, orientation = magnitude_orientation(gx, gy)
    sy, sx = image.shape
    orientations_image = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation)
    for j in range(1, dy):
        for i in range(1, dx):
            orientations_image[:-j, :-i, :] += orientations_image[j:, i:, :]
    
    if normalise:
        normalised_blocks = normalise_histogram(orientations_image, 1, 1, 1, 1, nbins)
    else:
        normalised_blocks = orientations_image
    
    if flatten:
        normalised_blocks = normalised_blocks.flatten()

    return normalised_blocks


def interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y, signed_orientation=False, nbins=9):
    """ Returns a matrix of size (cell_size_x, cell_size_y, nbins) corresponding
         to the trilinear interpolation of the pixels magnitude and orientation
    Parameters
    ----------
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.
    Returns
    -------
    orientation_histogram: array of shape (n_cells_x, n_cells_y, nbins)
            contains the histogram of orientation built using tri-linear interpolation
    """
    
    dx = csx//2
    dy = csy//2
    
    temp_coefs = interpolate_orientation(orientation, sx, sy, nbins, signed_orientation)


    # Coefficients of the spatial interpolation in every direction
    coefs = compute_coefs(csx, csy, dx, dy, n_cells_x, n_cells_y)
    
    temp = np.zeros((sy, sx, nbins))
    # hist(y0, x0)
    temp[:-dy, :-dx, :] += temp_coefs[dy:, dx:, :]*\
        (magnitude[dy:, dx:]*coefs[-(n_cells_y*csy - dy):, -(n_cells_x*csx - dx):])[:, :, np.newaxis]
    
    # hist(y1, x0)
    coefs = np.rot90(coefs)
    temp[dy:, :-dx, :] += temp_coefs[:-dy, dx:, :]*\
        (magnitude[:-dy, dx:]*coefs[:(n_cells_y*csy - dy), -(n_cells_x*csx - dx):])[:, :, np.newaxis]
    
    # hist(y1, x1)
    coefs = np.rot90(coefs)
    temp[dy:, dx:, :] += temp_coefs[:-dy, :-dx, :]*\
        (magnitude[:-dy, :-dx]*coefs[:(n_cells_y*csy - dy), :(n_cells_x*csx - dx)])[:, :, np.newaxis]
    
    # hist(y0, x1)
    coefs = np.rot90(coefs)
    temp[:-dy, dx:, :] += temp_coefs[dy:, :-dx, :]*\
        (magnitude[dy:, :-dx]*coefs[-(n_cells_y*csy - dy):, :(n_cells_x*csx - dx)])[:, :, np.newaxis]
    
    # Compute the histogram: sum over the cells
    orientation_histogram = temp.reshape((n_cells_y, csy, n_cells_x, csx, nbins)).sum(axis=3).sum(axis=1)
    
    return orientation_histogram


def draw_histogram(hist, csx, csy, signed_orientation=False):
    """ simple function to draw an orientation histogram
        with arrows
    """
    import matplotlib.pyplot as plt
    
    if signed_orientation:
        max_angle = 2*np.pi
    else:
        max_angle = np.pi
    
    n_cells_y, n_cells_x, nbins = hist.shape
    sx, sy = n_cells_x*csx, n_cells_y*csy
    plt.close()
    plt.figure()#figsize=(sx/2, sy/2))#, dpi=1)
    plt.xlim(0, sx)
    plt.ylim(sy, 0)
    center = csx//2, csy//2
    b_step = max_angle / nbins
        
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            for k in range(nbins):
                if hist[i, j, k] != 0:
                    width = 1*hist[i, j, k]
                    plt.arrow((center[0] + j*csx) - np.cos(b_step*k)*(center[0] - 1),
                              (center[1] + i*csy) + np.sin(b_step*k)*(center[1] - 1), 
                          2*np.cos(b_step*k)*(center[0] - 1), -2*np.sin(b_step*k)*(center[1] - 1),
                          width=width, color=str(width), #'black',
                          head_width=2.2*width, head_length=2.2*width,
                          length_includes_head=True)
                    
    plt.show()


def visualise_histogram(hist, csx, csy, signed_orientation=False):
    """ Create an image visualisation of the histogram of oriented gradient
    
    Parameters
    ----------
    hist: numpy array of shape (n_cells_y, n_cells_x, nbins)
        histogram of oriented gradient
    csx: int
        number of columns of the cells
    csy: int
        number of raws dimension of the cells
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    
    Return
    ------
    Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    """
    from skimage import draw
    
    if signed_orientation:
        max_angle = 2*np.pi
    else:
        max_angle = np.pi
    
    n_cells_y, n_cells_x, nbins = hist.shape
    sx, sy = n_cells_x*csx, n_cells_y*csy
    center = csx//2, csy//2
    b_step = max_angle / nbins

    radius = min(csx, csy) // 2 - 1
    hog_image = np.zeros((sy, sx), dtype=float)
    for x in range(n_cells_x):
        for y in range(n_cells_y):
            for o in range(nbins):
                centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                dx = radius * np.cos(o*nbins)
                dy = radius * np.sin(o*nbins)
                rr, cc = draw.line(int(centre[0] - dy),
                                   int(centre[1] - dx),
                                   int(centre[0] + dy),
                                   int(centre[1] + dx))
                hog_image[rr, cc] += hist[y, x, o]
    return hog_image


def normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins):
    """ normalises a histogram by blocks
    
    Parameters
    ----------
    bx: int
        number of blocks on the x axis
    by: int
        number of blocks on the y axis
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    nbins : int, optional, default is 9
        Number of orientation bins.
     
    The normalisation is done according to Dalal's original thesis, using L2-Hys.
    In other words the histogram is first normalised block-wise using l2 norm, before clipping it by
        limiting the values between 0 and 0.02 and finally normalising again with l2 norm
    """
    eps = 1e-7
    
    if bx==1 and by==1: #faster version
        normalised_blocks = np.clip(
          orientation_histogram / np.sqrt(orientation_histogram.sum(axis=-1)**2 + eps)[:, :, np.newaxis], 0, 0.2)
        normalised_blocks /= np.sqrt(normalised_blocks.sum(axis=-1)**2 + eps)[:, :, np.newaxis]
        
    else:
        n_blocksx = (n_cells_x - bx) + 1
        n_blocksy = (n_cells_y - by) + 1
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, nbins))

        for x in range(n_blocksx):
            for y in range(n_blocksy):
                block = orientation_histogram[y:y + by, x:x + bx, :]
                normalised_blocks[y, x, :] = np.clip(block[0, 0, :] / np.sqrt(block.sum()**2 + eps), 0, 0.2)
                normalised_blocks[y, x, :] /= np.sqrt(normalised_blocks[y, x, :].sum()**2 + eps)

    return normalised_blocks


def build_histogram(magnitude, orientation, cell_size=(8, 8), signed_orientation=False,
         nbins=9, cells_per_block=(1, 1), visualise=False, flatten=False, normalise=True):
    """ builds a histogram of orientation using the provided magnitude and orientation matrices
    
    Parameters
    ---------
    magnitude: np-array of size (sy, sx)
        matrix of magnitude
    orientation: np-array of size (sy, sx)
        matrix of orientations
    csx: int
        number of columns of the cells 
        MUST BE EVEN
    csy: int
        number of raws dimension of the cells
        MUST BE EVEN
    sx: int
        number of colums of the image (x axis)
    sy: int
        number of raws of the image (y axis)
    n_cells_x: int
        number of cells in the x axis
    n_cells_y: int
        number of cells in the y axis
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    nbins : int, optional, default is 9
        Number of orientation bins.
        
    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)
    
    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    """
    sy, sx = magnitude.shape
    csy, csx = cell_size
    
    # checking that the cell size are even
    if csx % 2 != 0:
        csx += 1
        print("WARNING: the cell_size must be even, incrementing cell_size_x of 1")
    if csy % 2 != 0:
        csy += 1
        print("WARNING: the cell_size must be even, incrementing cell_size_y of 1")
    
    # Consider only the right part of the image
    # (if the rest doesn't fill a whole cell, just drop it)
    sx -= sx % csx
    sy -= sy % csy
    n_cells_x = sx//csx
    n_cells_y = sy//csy
    magnitude = magnitude[:sy, :sx]
    orientation = orientation[:sy, :sx]
    by, bx = cells_per_block
    
    orientation_histogram = interpolate(magnitude, orientation, csx, csy, sx, sy, n_cells_x, n_cells_y, signed_orientation, nbins)
    
    if normalise:
        normalised_blocks = normalise_histogram(orientation_histogram, bx, by, n_cells_x, n_cells_y, nbins)
    else:
        normalised_blocks = orientation_histogram
    
    if flatten:
        normalised_blocks = normalised_blocks.flatten()

    if visualise:
        #draw_histogram(normalised_blocks, csx, csy, signed_orientation)
        return normalised_blocks, visualise_histogram(normalised_blocks, csx, csy, signed_orientation)
    else:
        return normalised_blocks


def histogram_from_gradients(gradientx, gradienty, cell_size=(8, 8), cells_per_block=(1, 1), signed_orientation=False,
        nbins=9, visualise=False, normalise=True, flatten=False, same_size=False):
    """ builds a histogram of oriented gradient from the provided gradients
    Parameters
    ----------
    gradientx : (M, N) ndarray
        Gradient following the x axis
    gradienty: (M, N) ndarray
        Gradient following the y axis
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    normalise: bool, optional, default is True
        if True, the histogram is normalised block-wise
    same_size: bool, optional, default is False
        if True, the boundaries are duplicated when computing the gradients of the image
        so that these have the same size as the original image
    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)
    
    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    magnitude, orientation = magnitude_orientation(gradientx, gradienty)
    return build_histogram(magnitude, orientation, cell_size=cell_size,
         signed_orientation=signed_orientation, cells_per_block=cells_per_block,
         nbins=nbins, visualise=visualise, normalise=normalise, flatten=flatten)


def hog(image, cell_size=(4, 4), cells_per_block=(1, 1), signed_orientation=False,
        nbins=9, visualise=False, normalise=True, flatten=False, same_size=True):
    """ builds a histogram of oriented gradient (HoG) from the provided image
    Compute a Histogram of Oriented Gradients (HOG) by
    1. computing the gradient image in x and y and deduce from them the magnitude and orientation
        of each pixel
    2. computing gradient histograms (vectorised version)
    3. normalising across blocks
    4. flattening into a feature vector if flatten=True
    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True
    signed_orientation: bool, default is True
        if True, sign information of the orientation is preserved,
            ie orientation angles are between 0 and 360 degree.
        if False, the angles are between 0 and 180 degree.
    normalise: bool, optional, default is True
        if True, the histogram is normalised block-wise
    same_size: bool, optional, default is True
        if True, the boundaries are duplicated when computing the gradients of the image
        so that these have the same size as the original image
    Returns
    -------
    if visualise if True: (histogram of oriented gradient, visualisation image)
    
    histogram of oriented gradient:
        numpy array of shape (n_cells_y, n_cells_x, nbins), flattened if flatten is True
    visualisation image:
        Image of shape (hist.shape[0]*csy, hist.shape[1]*csx)
    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    gx, gy = gradient(image, same_size=same_size)
    return histogram_from_gradients(gx, gy, cell_size=cell_size,
         signed_orientation=signed_orientation, cells_per_block=cells_per_block,
         nbins=nbins, visualise=visualise, normalise=normalise, flatten=flatten)


# gx, gy = gradient(img, same_size=False)
# mag, ori = magnitude_orientation(gx, gy)

# # Show gradient and magnitude
# plt.figure()
# plt.title('gradients and magnitude')
# plt.subplot(141)
# plt.imshow(img, cmap=plt.cm.Greys_r)
# plt.subplot(142)
# plt.imshow(gx, cmap=plt.cm.Greys_r)
# plt.subplot(143)
# plt.imshow(gy, cmap=plt.cm.Greys_r)
# plt.subplot(144)
# plt.imshow(mag, cmap=plt.cm.Greys_r)


# # Show the orientation deducted from gradient
# plt.figure()
# plt.title('orientations')
# plt.imshow(ori)
# plt.pcolor(ori)
# plt.colorbar()


# from scipy.ndimage.interpolation import zoom
# # make the image bigger to compute the histogram
# im1 = zoom(img, 3)
# h = hog(im1, cell_size=(2, 2), cells_per_block=(1, 1), visualise=False, nbins=9, signed_orientation=False, normalise=True)
# im2 = visualise_histogram(h, 8, 8, False)

# plt.figure()
# plt.title('HOG features')
# plt.imshow(im2, cmap=plt.cm.Greys_r)

# plt.show()






