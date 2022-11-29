import circle_fit as cf
import cv2
import imageio
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.io
import statistics

from utils import coord_from_arc_length, calc_arc_length
"""
RYAN
"""
def fluoro_get_coordinates(fluoro):
    """
    Given the fluoroscopy image, performs electrode registration
    and returns the coordinates for each registered electrode.
	Args:
		fluoro uint8 numpy array shape = (height,width) representing grayscale image
	Return:
        dictionary of coordinates for landmarks 
            ecog uint8 numpy array shape = (n,x,y) 
                where n is the number of electrodes
                x is the x coordinate of the electrode
                y is the y coordinate of the  electode
    """
    num_electrodes = 8

    # Load fluoroscopy image converted to greyscale
    gray = cv2.cvtColor(fluoro, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    # Guide at https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/
    params = cv2.SimpleBlobDetector_Params()
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 150                           # Best 150 > A > 700
    params.maxArea = 600
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.4                    # Best 0.4 > C > 0.8
    params.maxCircularity = 0.8
    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.7                      # Best 0.7 > I > 0.97
    params.maxConvexity = 0.97
    # Set Inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.1                   # Best 0.1 > I > 0.4
    params.maxInertiaRatio = 0.4
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    keypoints = detector.detect(gray)
    number_of_blobs = len(keypoints)
    ECoG_Coords = [keypoints[point].pt for point in range(0,len(keypoints))]
    ECoG_Coords_x = [coord[0] for coord in ECoG_Coords]
    ECoG_Coords_y = [coord[1] for coord in ECoG_Coords]
    ECoG_sep = math.dist(ECoG_Coords[0],ECoG_Coords[1])
    ave_ECoG_size = sum([keypoints[point].size for point in range(0,len(keypoints))])/len(keypoints)
    
    # Draw cv2 blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(gray, keypoints, blank, (255, 0, 0),
                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show best fit circle from https://github.com/AlliedToasters/circle-fit
    xc,yc,r,s = cf.least_squares_circle(ECoG_Coords)

    num_refits = 0
    while s > 150:       # Find the bad point(s)
        num_refits = num_refits + 1
        bad_indexes = []
        for i in range(len(keypoints)):
            ECoG_Coords_subset = ECoG_Coords[:i] + ECoG_Coords[i+1 :]
            
            xc_subset,yc_subset,r_subset,s_subset = cf.least_squares_circle(ECoG_Coords_subset)
            if s_subset < (s-1000):  # Keep the point
                bad_indexes.append(i)
        
        # Remake best fit circle
        ECoG_Coords = [i for j, i in enumerate(ECoG_Coords) if j not in bad_indexes]
        xc,yc,r,s = cf.least_squares_circle(ECoG_Coords)
        
        # Only attempt this a maximum of 4 times
        if num_refits > 3:
            break

    # If there are only 2 ecog electrodes this may find another but just hard codes y-axis drop
    if number_of_blobs == 2:
        # Predict the next electrode
        left_keypoint = keypoints[0] if keypoints[0].pt[0] < keypoints[1].pt[0] else keypoints[1]
        left_ECoG = [round(left_keypoint.pt[0]), round(left_keypoint.pt[1])]
        right_keypoint = keypoints[0] if keypoints[0].pt[0] > keypoints[1].pt[0] else keypoints[1]
        right_ECoG = [round(right_keypoint.pt[0]), round(right_keypoint.pt[1])]

        left_predict = [ECoG_Coords[0][0]-(ECoG_Coords[1][0]-ECoG_Coords[0][0]),
                        ECoG_Coords[0][1]-(ECoG_Coords[1][1]-ECoG_Coords[0][1])]
        left_predict_space = gray[math.floor(left_ECoG[1]-ave_ECoG_size*2):math.ceil(left_ECoG[1]+ave_ECoG_size*2),
                                math.floor(left_ECoG[0]-ave_ECoG_size*2):math.ceil(left_ECoG[0]+ave_ECoG_size*2)]
        right_predict = [ECoG_Coords[1][0]+(ECoG_Coords[1][0]-ECoG_Coords[0][0]),
                        ECoG_Coords[1][1]+(ECoG_Coords[1][1]-ECoG_Coords[0][1])]
        right_predict_space = gray[math.floor(right_ECoG[1]-ave_ECoG_size/2):math.ceil(right_ECoG[1]+ave_ECoG_size/2),
                                math.floor(right_ECoG[0]-ave_ECoG_size/2):math.ceil(right_ECoG[0]+ave_ECoG_size/2)]

        left_predict_shade = left_predict_space.mean(axis=0).mean(axis=0)
        right_predict_shade = right_predict_space.mean(axis=0).mean(axis=0)

        next_ECoG = left_predict if left_predict_shade > right_predict_shade else right_predict
        next_ECoG = [round(next_ECoG[0]), round(next_ECoG[1])+4]
        next_ECoG_space = fluoro[math.floor(next_ECoG[1]-ave_ECoG_size):math.ceil(next_ECoG[1]+ave_ECoG_size),
                                math.floor(next_ECoG[0]-ave_ECoG_size):math.ceil(next_ECoG[0]+ave_ECoG_size)]
        ECoG_Coords.append(tuple(next_ECoG))
        
        # Create new best fit circle
        xc,yc,r,s = cf.least_squares_circle(ECoG_Coords)
        
    # Get arc lengths along fit circle from circle bottom
    circle_bottom = [xc, yc+r]
    c = [xc,yc]  # Fit circle center point
    arc_lengths = []  # Initialize coordinate arc lengths
    for i in range(len(keypoints)):
        arc_lengths.append(calc_arc_length(circle_bottom, keypoints[i].pt, c))
        
    # Remove coordinates whose arc lengths are far away
    #  This may remove true coordinates on edges but they are easier to put back
    conf_coords = []
    arc_lengths_final = []
    ave = statistics.mean(arc_lengths)
    std = statistics.stdev(arc_lengths)
    for i in range(len(keypoints)):
        zscore = (arc_lengths[i]-ave)/std
        if zscore < 1.5:
            conf_coords.append(keypoints[i].pt)
            arc_lengths_final.append(arc_lengths[i])

    # Fill in holes
    if len(conf_coords) < num_electrodes:
        zipped = list(zip(arc_lengths_final,conf_coords))
        zipped.sort()
        sorted_coords = [truth for _,truth in zipped]
        sorted_arcs = [length for length,_ in zipped]
        diffs = [x-sorted_arcs[i-1] for i,x in enumerate(sorted_arcs)][1:]

        min_diff = min(diffs)
        new_arcs = []
        for i in range(1,len(sorted_arcs)):
            # If there is room for 2+ electrodes, add more between
            if min_diff*1.7 <= diffs[i-1]:
                fill_arcs = np.linspace(sorted_arcs[i-1], sorted_arcs[i], num=round(diffs[i-1]/min_diff)+1)[1:-1]
                new_arcs.append(fill_arcs)
        # Flatten list of lists
        new_arcs = [item for sublist in new_arcs for item in sublist]
        new_electrodes = [coord_from_arc_length(circle_bottom, c, arc_length) for arc_length in new_arcs]

        all_ECoG_Coords = [item for sublist in [conf_coords, new_electrodes] for item in sublist]

    # ----------

    # Edge detection
    dst = cv2.Canny(fluoro, 30, 150, None, 3)

    # Copy edges to the images that will display the results in BGR
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    cdstP2 = np.copy(cdstP)

    # Probabilistic Line Transform
    linesP = cv2.HoughLinesP(dst, 1, 1 * np.pi / 180, 50, None, 250, 20)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv2.LINE_AA)

    fluoro_size = [fluoro.shape[0], fluoro.shape[1]]

    # Find the lead line
    max_length = 0
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            start = linesP[i][0][0:2]
            end = linesP[i][0][2:4]
            angle = math.atan2(start[1]-end[1], start[0]-end[0])*180/math.pi
            length = math.dist(start, end)
            # Remove the vertical annotation line
            if (start[1] >= cdstP.shape[0]-1) and (end[1] == 0):
                continue
            # Remove lines that are not near-verticle
            if not 50 <= angle <= 130:
                continue
            # Remove lines that are not in the approx x-range
            if not (fluoro_size[1]*0.52 <= start[1] <= fluoro_size[1]*0.82) and (fluoro_size[1]*0.52 <= end[1] <= fluoro_size[1]*0.82):
                continue
            # Choose the longest line
            if length > max_length:
                max_length = length
                lead_line = [start, end]

            # Give filler line if not detected
            if linesP is None:
                lead_line = [[fluoro_size*0.62, fluoro_size*0.7],[fluoro_size*0.7,  fluoro_size*0.29]]

    pin_tips = np.array([[ 542., 1019.],
                        [1399.,  539.]])
    print('fluoro_segmentation.py successfully executed.')
    return {"ecog": all_ECoG_Coords,
            "dbs": lead_line,
            "pin": pin_tips }
