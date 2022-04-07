import numpy as np
class TrainingCameraParameters:

    CameraModel = "Blender virtual camera"

    K_Intrinsic_Matrix = np.array([[2905.0,    0.0, 1328.0],
                                   [   0.0, 2905.0,  747.0],
                                   [   0.0,    0.0,    1.0]])

    RT_Matrix = np.array([[ 0.4430, 0.8918,  0.0919, -3.1169],
                          [-0.0735, 0.1382, -0.9877,  3.4298],
                          [-0.8935, 0.4308,  0.1268, 22.8607]])

    P_Matrix = np.array([[ 100.2657, 3162.7773,   435.1946, 21304.5625],
                         [-880.8749,  723.3470, -2774.5020, 27040.6016],
                         [  -0.8935,    0.4308,     0.1268,    22.8607]])


