import sys, os
sys.path.append(os.path.abspath('./Utils'))
import cv2
import pickle
import Utils
import matplotlib.pyplot as plt
import ControlPoints
import CamerasProperties
import numpy as np
import REPPnPMatlab

with open('TrainingRecognitionPoints/TrainingResults/Training3DSiftDescriptors.pickle', 'rb') as handle:
    training_3D_sift_descriptors = pickle.load(handle)
kpsTrain = training_3D_sift_descriptors[0]
descsTrain = training_3D_sift_descriptors[1]
train2Dpts = training_3D_sift_descriptors[2]
train3Dpts = training_3D_sift_descriptors[3]


with open('TrainingRecognitionPoints/TrainingResults/Training3DSurfDescriptors.pickle', 'rb') as handle:
    training_3D_surf_descriptors = pickle.load(handle)

#Loading test image
image = cv2.imread("../ImagesDataSet/SyntheticTestImages/10.png")
#We keep the original image size for training image (1494x2656 px). We prefeer larger images for training.
test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#Loading test camera parameters 
TestCameraParameters = CamerasProperties.TestCameraParameters
K_test = TestCameraParameters.Virtual_cameras_intrinsic_parameters

#Getting Sift descriptors
sift = cv2.xfeatures2d.SIFT_create()
(kpsTest, descsTest) = sift.detectAndCompute(test_image, None)
test2Dpoints = np.array(Utils.get_2dpts_from_kps(kpsTest))

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descsTest, descsTrain)
matches = sorted(matches, key = lambda x:x.distance)
matches = matches[:50]

matched_train_indexes = [matches[i].trainIdx for i in range(len(matches))]
matched_train_3Dpoints = np.take(train3Dpts,matched_train_indexes,0)

matched_test_indexes = [matches[i].queryIdx for i in range(len(matches))]
matched_test_2Dpoints = np.take(test2Dpoints,matched_test_indexes,0)

dist_coeffs = np.zeros((4,1))

R, T, mask = REPPnPMatlab.REPPnP_matlab(matched_train_3Dpoints, matched_test_2Dpoints, K_test)
# success, rotation_vector, translation_vector = cv2.solvePnP(
#     matched_train_3Dpoints, 
#     #matched_test_2Dpoints, 
#     np.ascontiguousarray(matched_test_2Dpoints[:,:2]).reshape((-1,1,2)),
#     K_test, 
#     None, None, None, False, 
#     cv2.SOLVEPNP_EPNP)

# Computing rotation matrix
rotation_matrix = R
translation_vector = T
RT_test = np.column_stack([rotation_matrix,translation_vector])
#Visualizing result
ReferencePoints = ControlPoints.Control_Points
Points3D = ReferencePoints.Control_Points_3D
Points2D = [Utils.project_3d_point_to_2d_pixel(Points3D[i],K_test,RT_test) for i in range(len(Points3D))]

projected_image = Utils.draw_skeleton_object_from_points_array(image, Points2D,5)
Utils.show_image(projected_image)
temp =1 