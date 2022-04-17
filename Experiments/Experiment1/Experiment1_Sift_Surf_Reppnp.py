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
K_test = TestCameraParameters.Virtual_cameras_intrinsic_parameters

#Loading control 3d points
Control3DPoints = ControlPoints.Control_Points.Control_Points_3D[:12]

#Creating descriptors and matching objects
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

# #FLANN matching
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)

#BF matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

#Loading test images
test_images_dir = os.path.join(root_dir, "ImagesDataSet/SyntheticTestImages")
testImages_paths = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith('.png')]
testImages_paths = sorted(testImages_paths)

number_of_matches = np.arange(10, 200, 10)

#Loading ground thruth parameters
K_gt = CamerasProperties.TestCameraParameters.Virtual_cameras_intrinsic_parameters
RT_dict_gt = CamerasProperties.TestCameraParameters.Virtual_cameras_extrinsic_parameters

#Dictionary to store results
T_error_results_sift_reppnp = {}
R_error_results_sift_reppnp = {}
Reproj_error_results_sift_reppnp = {}
NumberOutliers_results_sift_reppnp = {}

T_error_results_surf_reppnp = {}
R_error_results_surf_reppnp = {}
Reproj_error_results_surf_reppnp = {}
NumberOutliers_results_surf_reppnp = {}

#initializing matlab engine
eng = matlab.engine.start_matlab()

for testImage_path in testImages_paths:
    #Loading test image
    image = cv2.imread(testImage_path)
    fileName = os.path.basename(testImage_path)
    #We keep the original image size for training image (1494x2656 px). We prefeer larger images for training.
    test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    #################################### Extracting SIFT descriptors ####################################

    (kpsTest, descsTest) = sift.detectAndCompute(test_image, None)
    test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

    #allmatches = flann.knnMatch(descsTest,Sift_descsTrain,k=1)
    #allmatches = sorted(allmatches, key = lambda x:x[0].distance)

    allmatches = bf.match(descsTest, Sift_descsTrain)
    allmatches = sorted(allmatches, key = lambda x:x.distance)

    #Initializing dictionary to store results
    T_error_results_sift_reppnp[fileName] = {}
    R_error_results_sift_reppnp[fileName] = {}
    Reproj_error_results_sift_reppnp[fileName] = {}
    NumberOutliers_results_sift_reppnp[fileName] = {}

    for nmatches in number_of_matches:        
        matches = allmatches[:nmatches]

        #matched_train_indexes = [matches[i][0].trainIdx for i in range(len(matches))]
        matched_train_indexes = [matches[i].trainIdx for i in range(len(matches))]
        matched_train_3Dpoints = np.take(Sift_train3Dpts,matched_train_indexes,0)

        #matched_test_indexes = [matches[i][0].queryIdx for i in range(len(matches))]
        matched_test_indexes = [matches[i].queryIdx for i in range(len(matches))]
        matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

        dist_coeffs = np.zeros((4,1))

        R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test, eng)
        
        #Storing results
        R_error_results_sift_reppnp[fileName][nmatches] = Utils.get_Rotational_error(RT_dict_gt[fileName][0:3,0:3],R)
        T_error_results_sift_reppnp[fileName][nmatches] = Utils.get_Tranlation_error(RT_dict_gt[fileName][:,3],T)
        Reproj_error_results_sift_reppnp[fileName][nmatches] = Utils.get_Reprojection_error(K_gt, RT_dict_gt[fileName], K_test, np.column_stack([R,T]), Control3DPoints)
        NumberOutliers_results_sift_reppnp[fileName][nmatches] = len(mask[mask==0])

    #################################### Extracting SURF descriptors ####################################

    (kpsTest, descsTest) = surf.detectAndCompute(test_image, None)
    test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

    #allmatches = flann.knnMatch(descsTest,Surf_descsTrain,k=1)
    #allmatches = sorted(allmatches, key = lambda x:x[0].distance)

    allmatches = bf.match(descsTest, Surf_descsTrain)
    allmatches = sorted(allmatches, key = lambda x:x.distance)

    #Initializing dictionary to store results
    T_error_results_surf_reppnp[fileName] = {}
    R_error_results_surf_reppnp[fileName] = {}
    Reproj_error_results_surf_reppnp[fileName] = {}
    NumberOutliers_results_surf_reppnp[fileName] = {}

    for nmatches in number_of_matches:        
        matches = allmatches[:nmatches]

        #matched_train_indexes = [matches[i][0].trainIdx for i in range(len(matches))]
        matched_train_indexes = [matches[i].trainIdx for i in range(len(matches))]
        matched_train_3Dpoints = np.take(Surf_train3Dpts,matched_train_indexes,0)

        #matched_test_indexes = [matches[i][0].queryIdx for i in range(len(matches))]
        matched_test_indexes = [matches[i].queryIdx for i in range(len(matches))]
        matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

        dist_coeffs = np.zeros((4,1))

        R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test, eng)
        
        #Storing results
        R_error_results_surf_reppnp[fileName][nmatches] = Utils.get_Rotational_error(RT_dict_gt[fileName][0:3,0:3],R)
        T_error_results_surf_reppnp[fileName][nmatches] = Utils.get_Tranlation_error(RT_dict_gt[fileName][:,3],T)
        Reproj_error_results_surf_reppnp[fileName][nmatches] = Utils.get_Reprojection_error(K_gt, RT_dict_gt[fileName], K_test, np.column_stack([R,T]), Control3DPoints)
        NumberOutliers_results_surf_reppnp[fileName][nmatches] = len(mask[mask==0])

eng.quit()

#Saving SIFT-REPPnP results
pd.DataFrame.from_dict(R_error_results_sift_reppnp).to_csv("Experiments/Experiment1/Results/results_sift_reppnp_R_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(T_error_results_sift_reppnp).to_csv("Experiments/Experiment1/Results/results_sift_reppnp_T_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(Reproj_error_results_sift_reppnp).to_csv("Experiments/Experiment1/Results/results_sift_reppnp_Reproj_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(NumberOutliers_results_sift_reppnp).to_csv("Experiments/Experiment1/Results/results_sift_reppnp_NumOutliers.csv", index = True, header=True) 

#Saving SURF-REPPnP results
pd.DataFrame.from_dict(R_error_results_surf_reppnp).to_csv("Experiments/Experiment1/Results/results_surf_reppnp_R_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(T_error_results_surf_reppnp).to_csv("Experiments/Experiment1/Results/results_surf_reppnp_T_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(Reproj_error_results_surf_reppnp).to_csv("Experiments/Experiment1/Results/results_surf_reppnp_Reproj_error.csv", index = True, header=True) 
pd.DataFrame.from_dict(NumberOutliers_results_surf_reppnp).to_csv("Experiments/Experiment1/Results/results_surf_reppnp_NumOutliers.csv", index = True, header=True) 

# #Loading test image
# image = cv2.imread("../ImagesDataSet/SyntheticTestImages/10.png")
# #We keep the original image size for training image (1494x2656 px). We prefeer larger images for training.
# test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

# #Loading test camera parameters 
# TestCameraParameters = CamerasProperties.TestCameraParameters
# K_test = TestCameraParameters.Virtual_cameras_intrinsic_parameters

# #Getting Sift descriptors
# sift = cv2.xfeatures2d.SIFT_create()
# (kpsTest, descsTest) = sift.detectAndCompute(test_image, None)
# test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

# #feature matching
# #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

# #matches = bf.match(descsTest, descsTrain)
# #matches = sorted(matches, key = lambda x:x.distance)
# #matches = matches[:50]

# #FLANN matching
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(descsTest,descsTrain,k=1)

# #matches = sorted(matches, key = lambda x:x[0].distance)
# #matches = matches[:50]

# matched_train_indexes = [matches[i][0].trainIdx for i in range(len(matches))]
# matched_train_3Dpoints = np.take(train3Dpts,matched_train_indexes,0)

# matched_test_indexes = [matches[i][0].queryIdx for i in range(len(matches))]
# matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

# dist_coeffs = np.zeros((4,1))

# R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test)
# # success, rotation_vector, translation_vector = cv2.solvePnP(
# #     matched_train_3Dpoints, 
# #     #matched_test_2Dpoints, 
# #     np.ascontiguousarray(matched_test_2Dpoints[:,:2]).reshape((-1,1,2)),
# #     K_test, 
# #     None, None, None, False, 
# #     cv2.SOLVEPNP_EPNP)

# # Computing rotation matrix
# rotation_matrix = R
# translation_vector = T
# RT_test = np.column_stack([rotation_matrix,translation_vector])
# #Visualizing result
# ReferencePoints = ControlPoints.Control_Points
# Points3D = ReferencePoints.Control_Points_3D
# Points2D = [Utils.project_3d_point_to_2d_pixel(Points3D[i],K_test,RT_test) for i in range(len(Points3D))]

# projected_image = Utils.draw_skeleton_object_from_points_array(image, Points2D,5)
# Utils.show_image(projected_image)
# temp =1 