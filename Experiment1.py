#Experiment 1
#For this experiment we will use synthetic dataset
#The main objective is to compare 4 different implementations
#Of the algorithm
#1) SIFT EPnP
#2) SIFT REPnP
#3) SURF EPnP
#4) SURF REPnP

import cv2

#Loading training images

detector = cv2.FeatureDetector_create("SIFT")


