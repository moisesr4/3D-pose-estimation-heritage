#Experiment 1
#For this experiment we will use synthetic dataset
#The main objective is to compare 4 different implementations
#Of the algorithm
#1) SIFT EPnP
#2) SIFT REPnP
#3) SURF EPnP
#4) SURF REPnP

import cv2
import pickle



detector = cv2.FeatureDetector_create("SIFT")


#Loading training data
with open('TrainingRecognitionPoints/TrainingResults/Training3DSiftDescriptors.pickle', 'rb') as handle:
    training_3D_sift_descriptors = pickle.load(handle)

with open('TrainingRecognitionPoints/TrainingResults/Training3DSurfDescriptors.pickle', 'rb') as handle:
    training_3D_surf_descriptors = pickle.load(handle)

