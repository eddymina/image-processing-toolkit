import image_processing as imp
from image_processing  import im_processing,color_adjust,segmentation,kernel
from image_processing.face_detection import detect_faces
import numpy as np 
import cv2


img = cv2.imread('image_processing/face_detection/test.jpg')
print('sdfds')


detected = detect_faces.detect(img,'image_processing/face_detection/deploy.prototxt.txt',
	'image_processing/face_detection/res10_300x300_ssd_iter_140000.caffemodel',.12)

imp.cv_plot(detected)



# import image_processing.im_processing as imp
# from image_processing.yolo import yolo


# img = cv2.imread('image_processing/images/img.jpg',cv2.COLOR_BGR2GRAY)
# print(img.shape)

# img = imp.resize(img,img.shape[0]//2,img.shape[1]//2)

# print(img.shape)


# warped = segmentation.convert_object(img)





# img= im_processing.cut(img,thresh=150)


# plt.figure(figsize= (30,30)
# imp.plot_grey(segmentation.otsu(img)[1])


# ims= [img,kernel.edge_filter(img)]
# imp.im_subplot(ims)



# ims = [img,color_adjust.grey_level_adjust(img,100)]

# imp.im_subplot(ims)

# imp.plot_grey(image)


# print(image.shape)
#yolo_path ='image_processing/yolo/'
# frame = yolo.predict(image, model = yolo_path + 'yolov2.weights', config= yolo_path +'yolov2.cfg')



# import matplotlib.pyplot as plt 
# import torch 

# time = np.arange(-np.pi,np.pi,.01)
# y = np.cos(time)

# print(y.shape)
# plt.plot(time,y)
# plt.show()







