import cv2
import imageio
from PIL import Image
import numpy as np
"""
MICAH
"""
def project_to_2d(postct_data, fluoro, pins_fl, pins_ct, coords_2d):
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

    # 1. Proprocess CT image(s) and pins coordinates

    # For each CT electrode coordinate, rotate the coorresponding slices 270 degrees,
    # and resize the rotated image to match the size of the fluoro image. 
    # Repeat this transformation process for the pin coordinates.
    sl_num = np.array([pins_ct[:,2]])
    sl_resized = []
    pins_ct2 = []
    for i in range(len(pins_ct)-1):
        # Transform CT images
        sl = np.rot90(np.rot90(np.rot90(postct_data[:, :, int(sl_num[0,i])])))
        sl2 = cv2.resize(sl,[fluoro.shape[0],fluoro.shape[1]])
        sl_resized.append(sl2)
    
        # Transform CT pin coordinates
        refArray = np.zeros([256,256])
        refArray[int(pins_ct[i,0]),int(pins_ct[i,1])] = 1e8
        refArray2 = np.rot90(np.rot90(np.rot90(refArray)))
        refArray3 = cv2.resize(refArray2,[fluoro.shape[0],fluoro.shape[1]])
        refImg = Image.fromarray(refArray3.T)
    
        # Find transformed CT pin coordinates
        ref = np.array(refImg).T
        xRef, yRef = np.unravel_index(np.argmax(ref), ref.shape)
        pins_ct2.append([xRef,yRef])
    
    sl_resized = np.array(sl_resized)
    pins_ct2 = np.array(pins_ct2)
    
    # 2. Find the affine 2x3 transformation matrix from the 3 landmark coordinates

    pins_fl = np.float32(pins_fl)
    pins_ct2 = np.float32(pins_ct2)
    Tr = cv2.getAffineTransform(pins_fl,pins_ct2)
    
    # 3. Perform affine transformation for fluoroscopic image and electrodes

    # Affine transform fluoroscopic image

    fluoro2 = cv2.warpAffine(fluoro,Tr,(fluoro.shape[0],fluoro.shape[1]))

    # Affine transform fluoroscopic electrode coordinates

    coords = []
    for i in range(len(coords_2d)):
        refArray = np.zeros([fluoro.shape[0],fluoro.shape[1]])
        refArray[int(coords_2d[i,0]),int(coords_2d[i,1])] = 1e8
        refArray2 = cv2.warpAffine(refArray,Tr,(fluoro.shape[0],fluoro.shape[1]))
        refImg = Image.fromarray(refArray2.T)
    
        # Find transformed electrode coordinates
        ref = np.array(refImg).T
        xRef, yRef = np.unravel_index(np.argmax(ref), ref.shape)
        coords.append([xRef,yRef])

    coords = np.array(coords)

    print('fluoro_ct_alignment.py successfully executed.')
    return coords
