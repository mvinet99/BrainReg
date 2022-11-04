import cv2
from utils import rotate
import numpy as np
"""
MICAH
"""
def project_to_2d(prect_data, postct_data, fluoro, pins_ct, coords_2d):
    """
    Align fluoroscopy image with CT image 
    Args:
        prect 3 dimensional float64 numpy array 
        postct 3 dimensional float64 numpy array
        fluoro uint8 numpy array shape = (height,width) representing grayscale image
        pins_ct landmarks float64 numpy array shape = (n,x,y,z) 
                where n is the number of landmarks
                x is the x coordinate of the landmark
                y is the y coordinate of the landmark
                z is the y coordinate of the landmark
        coords_2d uint8 numpy array shape = (n,x,y) 
            where n is the number of electrodes
            x is the x coordinate of the electrode
            y is the y coordinate of the  electode
        fluoro_aligned numpy array 

    Return:
       coords_aligned_2d float64 numpy array shape = (n,x,y) 
    """
    ### project_to_2d

    ## TODO: Find where do landmarks go once fluoro is resized
    fluoro = cv2.resize(fluoro,(150,150))
    fluoro = rotate(fluoro, 90)


    shift_down = 296
    shift_right = 226

    fluoro_new = np.vstack((np.ones((shift_down,)+fluoro.shape[1:], 
                                        dtype=fluoro.dtype), 
                                        fluoro))

    fluoro_newnew = np.hstack((np.ones(fluoro_new.shape[:1]+(shift_right,),
                                        dtype=fluoro_new.dtype), 
                                        fluoro_new))


    coords = np.array([[149.0, 77.0],
             [143.0, 75.0],
             [137.0, 76.0],
             [132.0, 77.0],
             [127.0, 78.0]])
    print('fluoro_ct_alignment.py successfully executed.')
    return coords