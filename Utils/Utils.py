import cv2
import matplotlib.pyplot as plt
import numpy as np
import copyreg

#OpenCV methods
def show_image(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def patch_Keypoint_pickiling(self):
    # See https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/11985056#11985056
    # Create the bundling between class and arguments to save for Keypoint class
    # See : https://stackoverflow.com/questions/50337569/pickle-exception-for-cv2-boost-when-using-multiprocessing/50394788#50394788
    def _pickle_keypoint(keypoint): #  : cv2.KeyPoint
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )
    # C++ Constructor, notice order of arguments : 
    # KeyPoint (float x, float y, float _size, float _angle=-1, float _response=0, int _octave=0, int _class_id=-1)

    # Apply the bundling to pickle
    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)



# This function return the X,Y,Z world coordinate point from a 
# x,y,z depth pixel point
#Inputs:
# pixel_coordinates(1x2)
# zdepth(1x1)
# K: Intrinsic Camera Parameters (fx,0,cx;0,fy,cy;0,0,1)
# RT: Extrinsic Camera Parameters
#Outputs:
# [X,Y,Z]: World Coordinate point
def from_xyzdepth_to_xyzworld(pixel_coordinates, zdepth, K, RT):
    R = RT[0:3,0:3]
    T = RT[0:3,3]
    R_transposed = np.transpose(R)
    newTranslation = - R_transposed @ T
    RTinv = np.column_stack([R_transposed,newTranslation])
    normalized_depth = zdepth * np.linalg.inv(K) @ np.transpose(np.hstack([pixel_coordinates,1]))
    world_3d_point = RTinv @ np.hstack([normalized_depth-T,1])

    return world_3d_point


def project_3d_point_to_2d_pixel(Point3D, K, RT):
    Point2D = K @ RT @ np.append(Point3D,1)
    nPoint2D = Point2D/Point2D[2]
    return nPoint2D[0:2]
