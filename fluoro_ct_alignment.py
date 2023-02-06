import cv2
import scipy
from PIL import Image
import numpy as np
import math
from utils import rotate
from utils import euclidean_distance_coords

def get_middle_half(postct_array):
  
  # Saggital axis (defined the z-direction) is the side of the head (rather than front-back, top-bottom)
  saggital_slices = postct_array.shape[2]
  first_quarter = int(saggital_slices * 1/4)
  third_quarter = int(saggital_slices * 3/4)

  # Use only the middle 1/2 of slices in the sagittal (z) direction
  middle_half = postct_array[:, :, np.s_[first_quarter:third_quarter]]
  middle_half_num = middle_half.shape[2]

  return(middle_half, middle_half_num)

def xySlope(start_crd, end_crd, z):
  xSlope = (end_crd[0] - start_crd[0]) / z
  ySlope = (end_crd[1] - start_crd[1]) / z
  return(xSlope, ySlope)


def endPointTraverse(xPos, yPos, xslope, yslope, numZ, ct_middle_array, ct_middle_num):
  if numZ == 0:
    return 0
  newX = xPos + xslope
  newY = yPos + yslope

  # Get updated value through traversing next slice
  current_pos = ct_middle_array[math.floor(yPos)][math.floor(xPos)][ct_middle_num - numZ]
  traversed_value = current_pos + endPointTraverse(newX, newY, xslope, yslope, numZ - 1, ct_middle_array, ct_middle_num)

  return(traversed_value)

def createNewImage(start_pos, distance_first, postct_array):
  # Run function to load CT as array and get middle half of CT in z-direction
  ct_middle_array, ct_middle_num = get_middle_half(postct_array)

  numY, numX, numZ = ct_middle_array.shape
  new2DMatrix = np.array([])

  for y in range(numY):
    newRow = np.array([])

    #get updated value in row
    for x in range(numX):
      xSlope, ySlope = xySlope(start_pos, (x, y), distance_first + numZ)
      #adjust x and y to coordinate at first slice
      startXAtFirstSlice = start_pos[0] + (xSlope * distance_first)
      startYAtFirstSlice = start_pos[1] + (ySlope * distance_first)

      traverse_results = endPointTraverse(startXAtFirstSlice, startYAtFirstSlice, xSlope, ySlope, numZ, ct_middle_array, ct_middle_num)
      newRow = np.append(newRow, [traverse_results])

    #add row to matrix
    if(new2DMatrix.size == 0):
      new2DMatrix = np.array([newRow])
    else:
      new2DMatrix = np.append(new2DMatrix, [newRow], axis = 0)

  # Rotate projected_2D image array by 90 degrees clockwise
  rot2d_Matrix = np.rot90(new2DMatrix, axes = (1,0))
  return(rot2d_Matrix)

def project3D_CT(postct_array):
  start_position = (128, 128)  # starting from the center minimizes distortion
  distFromImg = 256 * 1.5

  rotated_ct_matrix = createNewImage(start_position, distFromImg, postct_array)

  return rotated_ct_matrix

def transform(postct_data, fluoro, pins_fl, pins_ct, coords_2d):
    """
    Transform the fluoro image based on CT and fluoro landmarks
    
    """

    # Preprocess CT image(s) and pins coordinates

    # Subtract 340 from x direction to cut off GUI section of fluoro images
    coords_new = []
    for i in range(len(pins_fl)):
        coords_n = np.array([pins_fl[i,0]-340,pins_fl[i,1]])
        coords_new.append(coords_n)
    pins_fl = np.array(coords_new)

    fluorot = np.delete(fluoro, range(0,340),axis=1)
    
    # Subtract 340 from x direction for electrode coordinates
    
    coords_new = []
    for i in range(len(coords_2d)):
        coords_n = np.array([coords_2d[i,0]-340,coords_2d[i,1]])
        coords_new.append(coords_n)
    coords_2d = np.array(coords_new)

    # Account for cases where only one DBS pin tip point can be found: estimation of second pin tip point

    if len(pins_fl) == 3:

        # Transform fluoro pin coordinate
        refArray = np.zeros([fluorot.shape[0],fluorot.shape[1]])
        refArray[int(pins_fl[0,0]),int(pins_fl[0,1])] = 1e8
        refArray2 = rotate(refArray,180)
        refImg = Image.fromarray(refArray2.T)
    
        # Find transformed fluoro pin coordinate
        ref = np.array(refImg).T
        xRef, yRef = np.unravel_index(np.argmax(ref), ref.shape)

        point1 = [xRef,yRef]

        # Find the leftmost point of the two fluoro pin coordinates
        mn = np.amin([xRef, pins_fl[0,0]])

        # Append new coordinate to fluoro pins array
        if mn == xRef:
            pins_fl = np.array([[xRef,yRef],[pins_fl[0,0],pins_fl[0,1]],[pins_fl[1,0],pins_fl[1,1]],[pins_fl[2,0],pins_fl[2,1]]])
        else:
            pins_fl = np.array([[pins_fl[0,0],pins_fl[0,1]],[xRef,yRef],[pins_fl[1,0],pins_fl[1,1]],[pins_fl[2,0],pins_fl[2,1]]])

    

    # For now, take the middle point of the DBS lead start and end points as DBS lead coordinate

    pins_fl = np.array([[pins_fl[0,0],pins_fl[0,1]],[pins_fl[1,0],pins_fl[1,1]],[np.rint((pins_fl[2,0]+pins_fl[3,0])/2),np.rint((pins_fl[2,1]+pins_fl[3,1])/2)]])

    # Preprocess the CT images and pin/DBS lead coordinates to match fluoro images

    # Find the scaling factor
    img_shape = (postct_data.shape[0], postct_data.shape[1])
    reshaped_img_shape = (fluorot.shape[0], fluorot.shape[1])
    scale = np.divide(reshaped_img_shape, img_shape)
    
    # Apply the scaling factor
    pins_ct2 = []
    for i in range(len(pins_ct)):
    
        CT_new = np.multiply([pins_ct[i,0], pins_ct[i,1]], scale)
        pins_ct2.append(CT_new)

    pins_ct2 = np.array(pins_ct2)

    # 2. Find the transformation matrix from the 3 landmark coordinates, apply to fluoro image and resize

    # Find the transformation matrix
    rows, cols = fluorot.shape
    pins_fl = np.float32(pins_fl)
    pins_ct2 = np.float32(pins_ct2)
    M, mask = cv2.estimateAffinePartial2D(pins_fl, pins_ct2)

    # Perform the transformation to fluoroscopy image, resize image to match 256x256 CT image shape
    dst = cv2.warpAffine(fluorot, M, (rows, cols))
    dst2 = cv2.resize(dst,[256,256])
    
    # Apply transformation matrix and resize to all fluoro electrode coordinates to register them to CT image space

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
    
    return coords, dst2

# Input the postct_data, pins_ct2, pins_fluoro, fluoro image, rotation degrees, electrode coordinates, 
# and ground truth electrodes

def project_to_2d(postct_data, fluoro, pins_fl, pins_ct, coords_2d, aligned_coords_gt):

    # Define CT pins and CT pins image for rotation
    pinsct2 = np.array([pins_ct[1],pins_ct[3],pins_ct[4]])
    pins_image = np.zeros((postct_data.shape[0],postct_data.shape[1],postct_data.shape[2]))
    for i in range(len(pinsct2)):
        pins_image[pinsct2[i,0],pinsct2[i,1],pinsct2[i,2]] = 1

    # Define rotations in 3D space as follows:
    # (2,1) = rotation about the x-axis
    # (1,0) = rotation about the y-axis
    # (2,0) = rotation about the z-axis

    # Perform grid search of all possible rotation combinations, apply the best combination as final result
    euc_list = []
    config_list = []
    # x axis rotations
    xlist = [0, 90]
    for x in xlist:
        postct_data2 = scipy.ndimage.rotate(postct_data,x,axes=(2,1))
        pins_image2 = scipy.ndimage.rotate(pins_image,x,axes=(2,1))
        # y axis rotations
        ylist = [0]
        for y in ylist:
            postct_data3 = scipy.ndimage.rotate(postct_data2,y,axes=(1,0))
            pins_image3 = scipy.ndimage.rotate(pins_image2,y,axes=(1,0))
            # z axis rotations
            zlist = [0]
            for z in zlist:
                postct_data4 = scipy.ndimage.rotate(postct_data3,z,axes=(2,0))
                pins_image4 = scipy.ndimage.rotate(pins_image3,z,axes=(2,0))
                
                
                # Project to 2D
                projected_ct = project3D_CT(postct_data4)
                projected_pin_image = project3D_CT(pins_image4)

                # Retreive projected landmarks
                threshold, upper, lower = 0.1, 1, 0
                projected_pin_image = np.where(projected_pin_image>threshold, upper, lower)
                new_ct_pins = np.where(projected_pin_image == 1)

                new_ct_pins = np.concatenate(new_ct_pins)
                
                # Perform the transformation only if 3 landmarks were retrieved
                try:
                    # Define the correct order for the retreived CT landmarks
                    new_ct_pins = np.array([[new_ct_pins[5],new_ct_pins[2]],[new_ct_pins[3],new_ct_pins[0]],[new_ct_pins[4],new_ct_pins[1]]])

                    # Perform the transformation
                    coords, dst2 = transform(postct_data, fluoro, pins_fl, new_ct_pins, coords_2d)
                    
                    # Update configuration list
                    configuration = np.array([x,y,z])
                    config_list.append(configuration)
                    
                    # Calculate the Euclidean distance to ground truth coordinates
                    euc_dist = euclidean_distance_coords(coords, aligned_coords_gt)
                    euc_list.append(euc_dist)
                except:
                    # Update configuration list
                    configuration = np.array([x,y,z])
                    config_list.append(configuration)
                    
                    # Update Euclidean distance list
                    euc_dist = 10000
                    euc_list.append(euc_dist)

    # Find minimum Euclidean distance value and corresponding alignment configuration
    min_euc = np.argmin(np.array(euc_list))
    best_config = config_list[min_euc]

    # Apply the best configuration of rotations

    # x axis rotation
    postct_data2 = scipy.ndimage.rotate(postct_data,best_config[0],axes=(2,1))
    pins_image2 = scipy.ndimage.rotate(pins_image,best_config[0],axes=(2,1))

    # y axis rotation
    postct_data3 = scipy.ndimage.rotate(postct_data2,best_config[1],axes=(1,0))
    pins_image3 = scipy.ndimage.rotate(pins_image2,best_config[1],axes=(1,0))

    # z axis rotations
    postct_data4 = scipy.ndimage.rotate(postct_data3,best_config[2],axes=(2,0))
    pins_image4 = scipy.ndimage.rotate(pins_image3,best_config[2],axes=(2,0))
                        
    # Project to 2D
    projected_ct = project3D_CT(postct_data4)
    projected_pin_image = project3D_CT(pins_image4)

    # Retreive projected landmarks
    threshold, upper, lower = 0.1, 1, 0
    projected_pin_image = np.where(projected_pin_image>threshold, upper, lower)
    new_ct_pins = np.where(projected_pin_image == 1)
    new_ct_pins = np.concatenate(new_ct_pins)
                
    # Perform the transformation only if 3 landmarks were retrieved

    # Define the correct order for the retreived CT landmarks
    new_ct_pins = np.array([[new_ct_pins[5],new_ct_pins[2]],[new_ct_pins[3],new_ct_pins[0]],[new_ct_pins[4],new_ct_pins[1]]])

    # Perform the transformation
    coords, dst2 = transform(postct_data, fluoro, pins_fl, new_ct_pins, coords_2d)

    print('fluoro_ct_alignment.py successfully executed.')
    return coords