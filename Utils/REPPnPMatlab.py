import sys, os
import numpy as np
import matlab.engine

def  REPPnP_matlab(points_3d, points_2d, K):
    #Preparing data
    points_2d = np.transpose(np.hstack((points_2d, np.ones((points_2d.shape[0],1),dtype = points_2d.dtype))))
    u = np.linalg.lstsq(K,points_2d)
    u = u[0][0:2]

    U = np.transpose(points_3d)
    mU = U-np.mean(U,1).reshape(-1,1)

    #calling the matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(os.path.abspath('./Utils/REPPnP'), nargout=0)
    
    #[R, T, mask] = eng.REPPnP(mU,u)
    R_mat,T_mat,mask_inliers_mat = eng.REPPnP(matlab.double(mU.tolist()),matlab.double(u.tolist()), nargout=3)
    
    eng.quit()

    R = np.array(R_mat._data).reshape((3,3)).T
    T = np.asarray(T_mat)
    mask = np.asarray(mask_inliers_mat)
    #Converting result
    T = T - R @ np.mean(U,1).reshape(-1,1)
    return (R, T, mask)


