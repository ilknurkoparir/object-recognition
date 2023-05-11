# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 16:50:37 2023

@author: Lenovo
"""

# Importing Library
import cv2
import os
from glob import glob

jpeg=glob("D:/YOLO2/custom_yolo_model2/car_data/car_images/*.jpeg")
for j in jpeg:
    print(j)
    img=cv2.imread(j)
    cv2.imwrite(j[:-4]+"jpg",img)
    os.remove(j)







