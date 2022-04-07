import sys, os
sys.path.append(os.path.abspath('./Utils'))

import cv2
import Utils
import CamerasProperties

def generate_sift_training_descriptors(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    return (kps, descs)

def generate_surf_training_descriptors(gray_img):
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray, None)
    print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))
    return (kps, descs)


#Loading training image and depth image
image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.png")

#Loading depth image
hdr_image = cv2.imread("../ImagesDataSet/SyntheticTrainImage/VirtualImTrain.hdr", flags=cv2.IMREAD_ANYDEPTH)
depth_image = hdr_image[:,:,1]

#Loading training camera properties
TrainingCameraParameters = CamerasProperties.TrainingCameraParameters

#print(image.shape) -> (1494, 2656, 3)
#We keep the original image size for training images. 
#We just turn the image to gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Generating and storing SIFT descriptors points
(kps, descs) = generate_sift_training_descriptors(gray)
print(type(kps[0]))
print(type(descs))

#Depending on the descriptor type we get an array of [2D Pixel, 3D Point, DescriptorVector]