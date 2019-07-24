import image_processing as imp 
from image_processing import fourier_analysis,im_processing,color_adjust,kernel
import cv2
import numpy as np 
import matplotlib.pyplot as plt
###create an image with black lines  

#img = im_processing.cv_read('images/img_1.jpg','GRAY')

img = cv2.imread('images/img_9.jpg',1)

edged = cv2.Canny(img,100,200)


color_adjust.plot_hist(img)
#dim = tuple([int(.1 * i) for i in edged.shape[0:2]])
#edged = cv2.resize(edged , (dim[1],dim[0]), interpolation =cv2.INTER_AREA)

#imp.plot_grey(kernel.canny(img).isolate())

# def img_diag(img):
# 	return int((img.shape[0]**2+img.shape[1]**2)**(1/2))





# Hough accumulator array of theta vs rho
# accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
# 

# sin, cos = np.sin(thetas),np.cos(thetas)

# for i in range(len(edges_x)):
# 	for theta_i in range(len(thetas)):
# 		rho = int(round(edges_x[i] * cos[theta_i] + edges_y[i] * cos[theta_i])) + diag_len
# 		#accumulator[rho,theta_i] +=1 
print('ok')
# return accumulator
import math 

# 

def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
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
    diag_len= img_diag(img)

    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    sin, cos = np.sin(thetas),np.cos(thetas)
  
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

  
    edges= img > value_threshold if lines_are_white else img < value_threshold

   
    edges_y,edges_x= np.nonzero(edges)


    # Vote in the hough accumulator
    for i in range(len(edges_x)):
        x = edges_x[i]
        y = edges_y[i]

        for t_idx in range(len(thetas)):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos[t_idx] + y * sin[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos




def show_hough_line(img, accumulator, thetas, rhos):
    

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')
    plt.show()



# # accumulator, thetas, rhos = hough_line(edged)

# # show_hough_line(img, accumulator, thetas, rhos)


# lines = cv2.HoughLines(edged,1,np.pi/180, 200) 
  
# # The below for loop runs till r and theta values  
# # are in the range of the 2d array 
# for r,theta in lines[0]: 
      
#     # Stores the value of cos(theta) in a 
#     a = np.cos(theta) 
  
#     # Stores the value of sin(theta) in b 
#     b = np.sin(theta) 
      
#     # x0 stores the value rcos(theta) 
#     x0 = a*r 
      
#     # y0 stores the value rsin(theta) 
#     y0 = b*r 
      
#     # x1 stores the rounded off value of (rcos(theta)-1000sin(theta)) 
#     x1 = int(x0 + 1000*(-b)) 
      
#     # y1 stores the rounded off value of (rsin(theta)+1000cos(theta)) 
#     y1 = int(y0 + 1000*(a)) 
  
#     # x2 stores the rounded off value of (rcos(theta)+1000sin(theta)) 
#     x2 = int(x0 - 1000*(-b)) 
      
#     # y2 stores the rounded off value of (rsin(theta)-1000cos(theta)) 
#     y2 = int(y0 - 1000*(a)) 
      
#     # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). 
#     # (0,0,255) denotes the colour of the line to be  
#     #drawn. In this case, it is red.  
#     cv2.line(img,(x1,y1), (x2,y2), (0,0,255),2) 
      
# # All the changes made in the input image are finally 
# # written on a new image houghlines.jpg 
# imp.cv_plot(img)






