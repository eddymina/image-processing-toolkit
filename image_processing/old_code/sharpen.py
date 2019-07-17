import argparse
import im_processing as imp
import cv2 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Sharpening Algo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str,
                        help='Image Path')

    parser.add_argument('-l','--level', type=str, default="standard",
                        help='Level of sharpening:: "enhance" or standard" or "excessive"')



    img_path = parser.parse_args().input
    level = parser.parse_args().level

    # img_path = 'einstein.jpg'

    img = cv2.imread(img_path, cv2.IMREAD_COLOR )


    plt.suptitle('{} sharpening of {}'.format(level,img_path))
    t= ["Original {} Image".format(img.shape),"Sharpened Image"]

    if level == "enhance":
        s= imp.sharpen(img).edge_enhance()
    elif level == "standard":
        s= imp.sharpen(img).sharpen()
    elif level == "excessive":
        s= imp.sharpen(img).excessive()
    else:
        raise TypeError('Level of sharpening must be:: "enhance" or standard" or "excessive"')

    imp.im_subplot([img,s],shape=[1,2],titles= t) 




