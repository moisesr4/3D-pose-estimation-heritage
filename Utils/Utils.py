import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


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
