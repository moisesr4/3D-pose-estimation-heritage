import sys, os
sys.path.append(os.path.abspath('./Utils'))

import cv2
import Utils
import CamerasProperties
import pickle

#Loading training image
image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.png")
#We keep the original image size for training image (1494x2656 px). We prefeer larger images for training.
train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#Loading depth image
hdr_image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.hdr", flags=cv2.IMREAD_ANYDEPTH)
depth_image = hdr_image[:,:,1]

#Loading training camera properties
TrainingCameraParameters = CamerasProperties.TrainingCameraParameters
K = TrainingCameraParameters.K_Intrinsic_Matrix
RT = TrainingCameraParameters.RT_Matrix

#Generating and storing SIFT descriptors points
print("Generating SIFT descriptors")
(kps, descs) = Utils.generate_sift_training_descriptors(train_image)
(pts_2d, pts_3d) = Utils.get_2dpts_and_3dpts_from_kps(kps, depth_image, K, RT)

Utils.patch_Keypoint_pickiling(kps)
with open('TrainingRecognitionPoints/TrainingResults/Training3DSiftDescriptors.pickle', 'wb') as handle:
    pickle.dump((kps, descs, pts_2d, pts_3d), handle, protocol=pickle.HIGHEST_PROTOCOL)

#Generating and storing Surf descriptors points
print("Generating Surf descriptors")
(kps, descs) = Utils.generate_surf_training_descriptors(train_image)
(pts_2d, pts_3d) = Utils.get_2dpts_and_3dpts_from_kps(kps, depth_image, K, RT)

Utils.patch_Keypoint_pickiling(kps)
with open('TrainingRecognitionPoints/TrainingResults/Training3DSurfDescriptors.pickle', 'wb') as handle:
    pickle.dump((kps, descs, pts_2d, pts_3d), handle, protocol=pickle.HIGHEST_PROTOCOL)
