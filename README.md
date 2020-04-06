# OBJECT_DETECTION
Using deep learning YOLOv3 algorithm object detection is performed in Real time using OpenCV .
coco.names : It contains 80 different class names used in dataset that the algorithm is trained.
yolov3.cfg: The file contains the network configuration.
yolov3.weights: This file contains the pre-trained network's weights.

It uses the non-maximum suppression which removes redundant overlapping bounding boxes.Non maximum supression is controlled by a parameter nms Threshold.

OpenCV is used to open the webcam and draw the rectangle boxes on the objects that has been detected by the yolo algorithm
