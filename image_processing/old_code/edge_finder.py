import argparse
import image_processing
#.im_processing as imp
import im_processing as imp
import cv2 
import matplotlib.pyplot as plt 

def fast_finder(img):

    return imp.simple_kernel(img).edge_filter()

def isolate(img):

    return imp.edge_finder(img).isolate()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Canny Image Edge Detetection Algo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str,
                        help='Image Path')
    parser.add_argument('-s','--speed', type=str, default=None,
                        help='Speed of Edge Detection None or "fast"')

    img_path = parser.parse_args().input
    speed = parser.parse_args().speed

    # img_path = 'einstein.jpg'

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    if speed == 'fast':
    	i = fast_finder(img)
    else:
    	i = isolate(img)

    plt.suptitle('Canny Edge Detetection for {}'.format(img_path))
    t= ["Original {} Image".format(img.shape),"Image Edge"]
    imp.im_subplot([cv2.imread(img_path, cv2.IMREAD_COLOR),i],shape=[1,2],titles= t) 