import image_processing as imp 
from image_processing import fourier_analysis,im_processing

###create an image with black lines  
vert_lines = fourier_analysis.gen_line_im(size=(100,100),samp_freq=5,vert=True,balance = True) 
horiz_lines= fourier_analysis.gen_line_im(size=(100,100),samp_freq=5,vert=False,balance = True) 
checker= fourier_analysis.gen_line_im(size=(100,100),samp_freq=5,checker= True) 
rotated = im_processing.rot45(vert_lines)
box= fourier_analysis.gen_square_im(size=(100,100),n=2,val=1)


ims = [vert_lines,horiz_lines,checker,rotated,box]
ims = [fourier_analysis.im_noise(im).poisson() for im in ims]
titles= ['vert_lines','horiz_lines','checker','rotated45','square']

ims.extend([fourier_analysis.magnitude_spectrum(img) for img in ims])

titles.extend([t +' FFT' for t in titles])


imp.im_subplot(ims,shape=[2,5],titles=titles,
	suptitle='Understanding Fast Fourier Transforms (FFT)')





# ims= [i, spectrum(i,plot=False)]



