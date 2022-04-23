import sys, os
root_dir = os.path.abspath(".")
sys.path.append(os.path.join(root_dir, "Utils"))
import cv2
import pickle
import Utils
import matplotlib.pyplot as plt
import ControlPoints
import CamerasProperties
import numpy as np
import REPPnPMatlab
import matlab.engine
import pandas as pd
import time 

with open(os.path.join(root_dir, 'TrainingRecognitionPoints/TrainingResults/Training3DSiftDescriptors.pickle'), 'rb') as handle:
    training_3D_sift_descriptors = pickle.load(handle)
Sift_kpsTrain = training_3D_sift_descriptors[0]
Sift_descsTrain = training_3D_sift_descriptors[1]
Sift_train2Dpts = training_3D_sift_descriptors[2]
Sift_train3Dpts = training_3D_sift_descriptors[3]


with open(os.path.join(root_dir, 'TrainingRecognitionPoints/TrainingResults/Training3DSurfDescriptors.pickle'), 'rb') as handle:
    training_3D_surf_descriptors = pickle.load(handle)
Surf_kpsTrain = training_3D_surf_descriptors[0]
Surf_descsTrain = training_3D_surf_descriptors[1]
Surf_train2Dpts = training_3D_surf_descriptors[2]
Surf_train3Dpts = training_3D_surf_descriptors[3]

#Loading test camera parameters 
TestCameraParameters = CamerasProperties.TestCameraParameters
K_test = TestCameraParameters.Real_camera_intrinsic_parameters
dist_coeffs_test = TestCameraParameters.Real_camera_distortion_coefficients

#Creating descriptors and matching objects
sift = cv2.xfeatures2d.SIFT_create(2000)
surf = cv2.xfeatures2d.SURF_create(400) #Hessian threshold to get 2000 kpts (same as sift)

#FLANN matching
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

#Loading test images
#test_images_dir = os.path.join(root_dir, "ImagesDataSet/RealTestImages/NormalLightConditions")
test_images_dir = os.path.join(root_dir, "ImagesDataSet/RealTestImages/HardLightConditions")
testImages_paths = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
testImages_paths = sorted(testImages_paths)

number_of_matches = 100

#scaling image affects intrinsic parameters
factor_scale = 0.25
K_test = K_test*factor_scale
K_test[2][2] = 1

#initializing matlab engine
eng = matlab.engine.start_matlab()

for testImage_path in testImages_paths:
    fileName = os.path.basename(testImage_path)

    #Loading test image
    image = cv2.imread(testImage_path)
    #We scale the original image size for detection .
    image = cv2.resize(image, (0, 0), fx=factor_scale, fy=factor_scale)

    width = int(image.shape[1])
    height = int(image.shape[0])

    newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(K_test, dist_coeffs_test, (width, height), 1, (width, height))
    undistorted_image = cv2.undistort(image, K_test, dist_coeffs_test, None, newcameramatrix)
    
    test_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY) 

    #################################### Extracting SIFT descriptors ####################################

    (kpsTest, descsTest) = sift.detectAndCompute(test_image, None)
    test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

    allmatches = flann.knnMatch(descsTest,Sift_descsTrain,k=2)
    allmatches_filtered = Utils.filter_matches_lowe_distance(allmatches, 0.7)
    allmatches_filtered = sorted(allmatches_filtered, key = lambda x:x.distance)

    matches = allmatches_filtered[:number_of_matches]

    matched_train_indexes = [matches[i].trainIdx for i in range(len(matches))]
    matched_train_3Dpoints = np.take(Sift_train3Dpts,matched_train_indexes,0)

    matched_test_indexes = [matches[i].queryIdx for i in range(len(matches))]
    matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

    dist_coeffs = np.zeros((4,1))

    R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test, eng)

    #Storing results
    projected_image = Utils.Project_skeleton_on_image(undistorted_image, K_test,R,T)
    #file_path = "Experiments/Experiment2/Outputs/SIFT_REPPnP/NLC_" + fileName
    file_path = "Experiments/Experiment2/Outputs/SIFT_REPPnP/HLC_" + fileName
    cv2.imwrite(file_path, cv2.resize(projected_image, (0, 0), fx=0.5, fy=0.5))

    #################################### Extracting SURF descriptors ####################################

    start_time_matching_descriptors = time.time() #timinig

    (kpsTest, descsTest) = surf.detectAndCompute(test_image, None)
    test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

    allmatches = flann.knnMatch(descsTest,Surf_descsTrain,k=2)
    allmatches_filtered = Utils.filter_matches_lowe_distance(allmatches, 0.7)
    allmatches_filtered = sorted(allmatches_filtered, key = lambda x:x.distance)
   
    matches = allmatches_filtered[:number_of_matches]

    matched_train_indexes = [matches[i].trainIdx for i in range(len(matches))]
    matched_train_3Dpoints = np.take(Surf_train3Dpts,matched_train_indexes,0)

    matched_test_indexes = [matches[i].queryIdx for i in range(len(matches))]
    matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

    dist_coeffs = np.zeros((4,1))

    R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test, eng)
    
    #Storing results
    projected_image = Utils.Project_skeleton_on_image(undistorted_image, K_test,R,T)
    #file_path = "Experiments/Experiment2/Outputs/SURF_REPPnP/NLC_" + fileName
    file_path = "Experiments/Experiment2/Outputs/SURF_REPPnP/HLC_" + fileName
    cv2.imwrite(file_path, cv2.resize(projected_image, (0, 0), fx=0.5, fy=0.5))

eng.quit()