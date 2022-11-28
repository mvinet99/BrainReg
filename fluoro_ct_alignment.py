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
    # Subtract 340 from x direction to cut off GUI section of fluoro images
    pins_flx = np.array([x - 340 for x in pins_fl[:,0]])
    pins_fl = np.array([[pins_flx[0],pins_fl[0,1]], [pins_flx[1],pins_fl[1,1]], [pins_flx[2], pins_fl[2,1]]])
    fluorot = np.delete(fluoro, range(0,340),axis=1)
    coords_new = []
    for i in range(len(coords_2d)):
        coords_n = np.array([coords_2d[i,0]-340,coords_2d[i,1]])
        coords_new.append(coords_n)
    coords_2d = np.array(coords_new)
    # Define the CT pins and DBS lead coordinates from inputs
    pins_ct = np.array([pins_ct[1],pins_ct[3],pins_ct[4]])

    # 1. Preprocess the CT images and pin/DBS lead coordinates to match fluoro images

    # Find the scaling factor
    img_shape = (postct_data.shape[0], postct_data.shape[1])
    reshaped_img_shape = (fluorot.shape[0], fluorot.shape[1])
    scale = np.divide(reshaped_img_shape, img_shape)

    sl_num = np.array([pins_ct[:,2]])
    sl_resized = []
    pins_ct2 = []
    for i in range(len(pins_ct)):
        # Transform CT images
        sl = postct_data[:, :, int(sl_num[0,i])]
        sl2 = cv2.resize(sl,[fluorot.shape[0],fluorot.shape[1]])
    
        CT_new = np.multiply([pins_ct[i,0], pins_ct[i,1]], scale)
        pins_ct2.append(CT_new)

    sl_resized = np.array(sl_resized)
    pins_ct2 = np.array(pins_ct2)

    # For CT DBS lead coordinate and fluoro DBS lead coordinate, move in the x- and y-axes to form the proper triangle for transformation

    pins_fl = np.array([[pins_fl[0,0],pins_fl[0,1]],[pins_fl[1,0],pins_fl[1,1]],[pins_fl[2,0],pins_fl[2,1]-400]])
    pins_ct2 = np.array([[pins_ct2[0,0]+220,pins_ct2[0,1]+420],[pins_ct2[1,0]+220,pins_ct2[1,1]+230],[pins_ct2[2,0]+320,pins_ct2[2,1]-70]])

    # 2. Find the affine 2x3 transformation matrix from the 3 landmark coordinates, apply to fluoro image and resize

    # Find 2x3 affine transformation matrix
    rows, cols = fluorot.shape
    pins_fl = np.float32(pins_fl)
    pins_ct2 = np.float32(pins_ct2)
    M = cv2.getAffineTransform(pins_fl, pins_ct2)

    # Perform 2x3 affine transformation to fluoroscopy image, resize image to match 256x256 CT image shape
    dst = cv2.warpAffine(fluorot, M, (rows, cols))
    dst2 = cv2.resize(dst,[sl.shape[0],sl.shape[1]])
    
    # 3. Apply transformation matrix and resize to all fluoro electrode coordinates to register them to CT image space

    # Find the scaling factor
    img_shape = (dst2.shape[1], dst2.shape[0])
    reshaped_img_shape = (dst.shape[1], dst.shape[0])
    scale = np.divide(img_shape, reshaped_img_shape)

    # Apply transformation matrix
    coords = []
    for i in range(len(coords_2d)):
        pt = coords_2d[i]
        new_x = M[0,0]*pt[0] + M[0,1]*pt[1] + M[0,2]
        new_y = M[1,0]*pt[0] + M[1,1]*pt[1] + M[1,2]
    
        # Find resized coordinates in CT image space
        coord_new = np.multiply([new_x, new_y], scale)
        coords.append(coord_new)

    coords = np.array(coords)

    print('fluoro_ct_alignment.py successfully executed.')
    return coords