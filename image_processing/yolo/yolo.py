import cv2
import argparse


def predict(image,model='yolov2.weights',config= 'yolov2.cfg'):
  # Minimum confidence threshold. Increasing this will improve false positives but will also reduce detection rate.
  min_confidence=0.14

  #Load names of classes
  classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 
   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
   'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
     'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
     'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
     'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
     'hair drier', 'toothbrush']


  # Load weights and construct graph
  net = cv2.dnn.readNetFromDarknet(config, model)
  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
  net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


  #Read input image
  frame = image 

  # Get width and height
  height,width,ch=frame.shape

  # Create a 4D blob from a frame.
  blob = cv2.dnn.blobFromImage(frame, 1.0/255.0, (416, 416), True, crop=False)
  net.setInput(blob)


  # Run the preprocessed input blog through the network
  predictions = net.forward()
  probability_index=5

  for i in range(predictions.shape[0]):
      prob_arr=predictions[i][probability_index:]
      class_index=prob_arr.argmax(axis=0)
      confidence= prob_arr[class_index]
      if confidence > min_confidence:
          x_center=predictions[i][0]*width
          y_center=predictions[i][1]*height
          width_box=predictions[i][2]*width
          height_box=predictions[i][3]*height
       
          x1=int(x_center-width_box * 0.5)
          y1=int(y_center-height_box * 0.5)
          x2=int(x_center+width_box * 0.5)
          y2=int(y_center+height_box * 0.5)
       
          cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),1)
          cv2.putText(frame,classes[class_index]+" "+"{0:.1f}".format(confidence),(x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA)
          # cv2.imwrite("out_"+args.input, frame)
   
  return frame

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='Path to input image.')
  args = parser.parse_args()
  i= args.input

 


  frame =predict(cv2.imread(i))
  cv2.imshow('img',frame)
  cv2.waitKey()
  cv2.destroyAllWindows()
  # if (cv2.waitKey() >= 0):
  #     cv2.destroyAllWindows(1)




















