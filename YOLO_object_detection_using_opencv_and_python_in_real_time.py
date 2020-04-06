#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 23:29:31 2020

@author: hrithik
"""


import cv2
import numpy as np


#load yolo

font=cv2.FONT_HERSHEY_PLAIN
net=cv2.dnn.readNet("/home/hrithik/Desktop/YOLO/yolov3.weights","/home/hrithik/Desktop/YOLO/yolov3.cfg")
classes=[]
with open("/home/hrithik/Desktop/YOLO/coco.names","r") as f: classes=[line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
colors=np.random.uniform(0,255,size=(len(classes),3))

#loading img

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
#img=cv2.imread("/home/hrithik/Desktop/test.jpg")
#to resize the image
#img=cv2.resize(img,None,fx=0.4,fy=0.4)
    #height,width,channels=img.shape
    height,width,channels=frame.shape
    
    #blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True , crop=False)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True , crop=False)
    
    """
    for b in blob:
        for n,img_blob in enumerate(b,0):
            cv2.imshow(str(n),img_blob)
    """
    net.setInput(blob)
    outs=net.forward(output_layers)
    #print(outs)
    
    class_ids=[]
    confidences=[]
    boxes = []
    
    
    #showing information on the screen
    for out in outs:
        for detection in out:
            scores= detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence > 0.5:
                #object detected
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                
                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                
                #rectangle coordinates
                x=int(center_x - w / 2)
                y=int(center_y - h / 2)
            
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255))
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    #print(len(boxes))
    number_object_detected =len(boxes)
    indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    
    for i in range (len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            #print(label)
            color=colors[i]
            #cv2.rectangle(img, (x,y),(x+w, y+h),color,2)
            #cv2.putText(img, label, (x, y + 30), font, 3,color,2)
            cv2.rectangle(frame, (x,y),(x+w, y+h),color,2)
            cv2.putText(frame, label, (x, y + 30), font, 3,color,2)
        
        
    #cv2.imshow("Image",img)
    cv2.imshow("Image",frame)
    #cv2.waitKey(0)
    key=cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()
