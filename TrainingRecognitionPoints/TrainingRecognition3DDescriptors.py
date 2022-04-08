import sys, os
sys.path.append(os.path.abspath('./Utils'))

import cv2
import Utils
import CamerasProperties
import pickle

def generate_sift_training_descriptors(gray_img):
    print("Generating SIFT descriptors")
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray_img, None)
    return (kps, descs)

def generate_surf_training_descriptors(gray_img):
    print("Generating Surf descriptors")
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray_img, None)
    return (kps, descs)


#Loading training image
image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.png")
#We keep the original image size for training image (1494x2656 px). We prefeer larger images for trainig 
train_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

#Loading depth image
hdr_image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.hdr", flags=cv2.IMREAD_ANYDEPTH)
depth_image = hdr_image[:,:,1]

#Loading training camera properties
TrainingCameraParameters = CamerasProperties.TrainingCameraParameters
K = TrainingCameraParameters.K_Intrinsic_Matrix
RT = TrainingCameraParameters.RT_Matrix

#Generating and storing SIFT descriptors points
(kps, descs) = generate_sift_training_descriptors(train_image)

pts_2d = [kps[i].pt for i in range(len(kps))]
z_depths = [depth_image[round(pts_2d[i][1])][round(pts_2d[i][0])] for i in range(len(pts_2d))]
pts_3d = [Utils.from_xyzdepth_to_xyzworld(pts_2d[i], z_depths[i], K, RT) for i in range(len(pts_2d))]

Utils.patch_Keypoint_pickiling(kps)
with open('filename.pickle', 'wb') as handle:
    pickle.dump((kps, descs, pts_2d, pts_3d), handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
