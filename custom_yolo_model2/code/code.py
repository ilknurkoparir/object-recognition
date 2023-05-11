# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:13:50 2023

@author: Lenovo
"""

#%% 1.Bolum
import cv2
import numpy as np

img = cv2.imread("D:/YOLO2/pretrained_image2/image/people2.jpg")
print(img)

img_width=img.shape[1]
img_height=img.shape[0]
 #%% 2. Bolum
img_blob = cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
labels = ["car"]

#Labellar icin renk olusturalim
colors=["0,0,255","229,219,58","255,52,184","0,0,0","255,255,150"]
colors=[np.array(color.split(",")).astype("int") for color in colors]
colors=np.array(colors)
colors=np.tile(colors,(18,1))

#%% 3. Bolum
#Algoritmamizi dahil edelim
model=cv2.dnn.readNetFromDarknet("D:/YOLO2/custom_yolo_model2/yolov4/darknet.zip/darknet/yolov4.cfg","D:/YOLO2/custom_yolo_model2/yolov4/darknet.zip/darknet/yolov4.weights")
layers=model.getLayerNames()
output_layer = [layers[layer - 1] for layer in model.getUnconnectedOutLayers()]
model.setInput(img_blob)
detection_layers = model.forward(output_layer)
###### NON-MAXIMUM-OPERATİON 1 ###########

id_list=[]
boxes_list=[]
confidences_list=[]


######END-OF-OPERATİON ###########
#%% 4. Bolum
for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores = object_detection[5:]
        predicted_id=np.argmax(scores)
        confidence=scores[predicted_id]
        
        if confidence > 0.90:
            label=labels[predicted_id]
            bounding_box=object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
            (box_center_x,box_center_y,box_width,box_height)=bounding_box.astype("int")
            
            start_x = int(box_center_x-(box_width/2))
            start_y = int(box_center_y-(box_height/2))
            
            ###### NON-MAXIMUM-OPERATİON 2 ###########
            
            id_list.append(predicted_id)
            confidences_list.append(float(confidence))
            boxes_list.append(start_x,start_y,int(box_width),int(box_height))
            
            
            ######END-OF-OPERATİON ###########
            
            
 ######## NON-MAXIMUM-OPERATİON 3 ###########
 
 
 
 
            
######END-OF-OPERATİON ###########
    
            end_x=start_x+box_width
            end_y=start_y+box_height
            box_color=colors[predicted_id]
            box_color=[int(each)for each in box_color]
            
            label="{}:{:.2f}%".format(label,confidence*100)
            print("predicted object {}".format(label))
            
            cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,1)
            cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
            
            
            
            
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()