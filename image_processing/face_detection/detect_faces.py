import cv2
import numpy as np

# load our serialized model from disk
def detect(image,prototxt, model,conf=.14):
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(prototxt, model)

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > conf:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	 
			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output image
	return image 
	
img = cv2.imread('test.jpg',cv2.COLOR_BGR2GRAY)


def cv_cascade(img, xml= 'haarcascade_frontalface_alt2.xml'):
	face_cascade = cv2.CascadeClassifier(xml)
	faces = face_cascade.detectMultiScale(img, 1.01, 5)

	for (x,y,w,h) in faces:
	    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow(' ',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# i = detect(img, 'deploy.prototxt.txt','res10_300x300_ssd_iter_140000.caffemodel')

# import matplotlib.pyplot as plt 
# plt.imshow(i)

# plt.show()





