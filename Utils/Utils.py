import cv2
import matplotlib.pyplot as plt
import numpy as np
import copyreg
import csv
import ControlPoints
from scipy.spatial.transform import Rotation

#OpenCV methods
def get_Tranlation_error(Tref,T):
    return (np.linalg.norm(Tref - T)/np.linalg.norm(T))*100

def get_Rotational_error(Rref, R):
    Qref = Rotation.from_matrix(Rref).as_quat()
    Q = Rotation.from_matrix(R).as_quat()
    return (np.linalg.norm(Qref - Q)/np.linalg.norm(Q))*100
    # error_mat = Rref @ R.T * -1
    # diff_rotation_vector = np.zeros(shape=3)
    # cv2.Rodrigues(error_mat, diff_rotation_vector)
    # return np.linalg.norm(diff_rotation_vector)/3

def get_Reprojection_error(Kref, RTref, K, RT, CtrPts3D):
    Points2Dref = [project_3d_point_to_2d_pixel(CtrPts3D[i], Kref, RTref) for i in range(len(CtrPts3D))]
    Points2D = [project_3d_point_to_2d_pixel(CtrPts3D[i], K, RT) for i in range(len(CtrPts3D))]
    diff_Points2d = np.subtract(Points2Dref, Points2D)
    norm_diff_Points2d = [np.linalg.norm(diff_Points2d[i]) for i in range(len(diff_Points2d))]
    return max(norm_diff_Points2d)/2

def get_NumOutliers_from_Inliers(matches, inliers):
    if inliers is not None:
        return len(matches) - len(inliers)
    return len(matches)


def show_image(image):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def draw_line_from_points_array(image, points, line_thickness = 2):
    drawnImage = image.copy()
    for i in range(len(points)-1):
        cv2.line(drawnImage, (round(points[i][0]), round(points[i][1])), (round(points[i+1][0]), round(points[i+1][1])), (0, 255, 0), thickness=line_thickness)
    return drawnImage

def draw_closed_line_from_points_array(image, points, line_thickness = 2):
    drawnImage = draw_line_from_points_array(image, points, line_thickness)
    cv2.line(drawnImage, (round(points[-1][0]), round(points[-1][1])), (round(points[0][0]), round(points[0][1])), (0, 255, 0), thickness=line_thickness)
    return drawnImage

def draw_skeleton_object_from_points_array(image, points, line_thickness = 2):
    drawnImage = draw_closed_line_from_points_array(image, points[:12], line_thickness)
    cv2.line(drawnImage, (round(points[5][0]), round(points[5][1])), (round(points[11][0]), round(points[11][1])), (0, 255, 0), thickness=line_thickness)
    cv2.line(drawnImage, (round(points[6][0]), round(points[6][1])), (round(points[10][0]), round(points[10][1])), (0, 255, 0), thickness=line_thickness)
    cv2.line(drawnImage, (round(points[4][0]), round(points[4][1])), (round(points[12][0]), round(points[12][1])), (0, 255, 0), thickness=line_thickness)
    
    #drawing projection lines
    cv2.line(drawnImage, (round(points[5][0]), round(points[5][1])), (round(points[13][0]), round(points[13][1])), (0, 255, 0), thickness=line_thickness)
    cv2.line(drawnImage, (round(points[7][0]), round(points[7][1])), (round(points[13][0]), round(points[13][1])), (0, 255, 0), thickness=line_thickness)
    #drawline(drawnImage, (round(points[8][0]), round(points[8][1])), (round(points[14][0]), round(points[14][1])), (0, 255, 0), thickness=line_thickness)
    #drawline(drawnImage, (round(points[13][0]), round(points[13][1])), (round(points[14][0]), round(points[14][1])), (0, 255, 0), thickness=line_thickness)

    return drawnImage

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

#descriptors
def generate_sift_training_descriptors(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray_img, None)
    return (kps, descs)

def generate_surf_training_descriptors(gray_img):
    surf = cv2.xfeatures2d.SURF_create()
    (kps, descs) = surf.detectAndCompute(gray_img, None)
    return (kps, descs)

def get_2dpts_and_3dpts_from_kps(kps, depth_image, K, RT):
    pts_2d = get_2dpts_from_kps(kps)
    pts_3d = get_3dpts_from_2dpts_and_depth_image(pts_2d, depth_image, K, RT)
    return (pts_2d, pts_3d)

def get_3dpts_from_2dpts_and_depth_image(pts_2d, depth_image, K, RT):
    z_depths = [depth_image[round(pts_2d[i][1])][round(pts_2d[i][0])] for i in range(len(pts_2d))]
    pts_3d = [from_xyzdepth_to_xyzworld(pts_2d[i], z_depths[i], K, RT) for i in range(len(pts_2d))]
    return pts_3d

def get_2dpts_from_kps(kps):
    return [kps[i].pt for i in range(len(kps))]



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
    #RTinv = np.column_stack([R_transposed,newTranslation])
    normalized_depth = zdepth * np.linalg.inv(K) @ np.transpose(np.hstack([pixel_coordinates,1]))
    world_3d_point = np.linalg.inv(R) @ (normalized_depth-T)

    return world_3d_point

def project_3d_point_to_2d_pixel(Point3D, K, RT):
    Point2D = K @ RT @ np.append(Point3D,1)
    nPoint2D = Point2D/Point2D[2]
    return nPoint2D[0:2]

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def write_dictionary_to_csv(path_to_write, dict):
    with open(path_to_write, 'w') as f:
        w = csv.DictWriter(f, dict.keys())
        w.writeheader()
        w.writerow(dict)

def filter_matches_lowe_distance(matches, ratio_thresh):
    goodMatches = []
    for match in matches:
        if match[0].distance < ratio_thresh * match[1].distance :
            goodMatches.append(match[0])
    return goodMatches

def EPnP_Ransac(matched_train_3Dpoints, matched_test_2Dpoints, K_test):
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
    matched_train_3Dpoints, 
    np.ascontiguousarray(matched_test_2Dpoints[:,:2]).reshape((-1,1,2)),
    K_test, 
    None, None, None, False, 
    200, 4, 0.99,
    cv2.SOLVEPNP_EPNP)

    # Computing rotation matrix
    rotation_matrix = np.zeros(shape=(3,3))
    cv2.Rodrigues(rvecs, rotation_matrix)

    return rotation_matrix, tvecs, inliers

def Project_skeleton_on_image(img, K, R, T):
    # Computing rotation matrix
    RT = np.column_stack([R,T])
    #Visualizing result
    ReferencePoints = ControlPoints.Control_Points
    Points3D = ReferencePoints.Control_Points_3D
    Points2D = [project_3d_point_to_2d_pixel(Points3D[i],K,RT) for i in range(len(Points3D))]

    projected_image = draw_skeleton_object_from_points_array(img, Points2D,5)
    return projected_image