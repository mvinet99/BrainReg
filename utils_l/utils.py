from matplotlib import animation, rc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import os
import itertools
# get four max points of a 3d array
def get_four_max_points(inp):
    """
    Function to get four max points of a 3d array
    Args
        array np.ndarray floainp
    Returns
        four_max_points np.ndarray float
    """
    array = np.copy(inp)
    four_max_points = np.zeros((4,3))
    for i in range(4):
        idx = np.unravel_index(np.argmax(array, axis=None), array.shape)
        four_max_points[i] = idx
        array[idx[0], idx[1], idx[2]] = 0
    return four_max_points
# get four max points of a 3d tensor tensorflow


# get cooridnates of all points on hull
def video_to_points(hull_data):
    """
    Function to find coordinates of all points on a hull
    Args
        hull_data np.ndarray float
    Returns
        coord_list np.ndarray float

    This function returns a list of coordinates of all the 
    points on the input hull data.

    """

    return np.vstack(np.nonzero(hull_data)).transpose(1,0)

## sort a list of 3d coordinates based on distance from the origin
def sort_coords(coords):
    """
    Function to sort a list of 3d coordinates based on distance from the origin
    Args
        coords np.ndarray float
    Returns
        coords np.ndarray float
    """
    coords = coords.tolist()
    coords.sort(key=lambda x: dist(x, np.array([0,0,0])))
    return np.array(coords)

def from_hull_to_ct_coords(ct_coords, nifti):
    """
    Converts the coordinates from the hull to the coordinates of the CT scan
    Args:
        ct_coords np.ndarray float (N,3)
        nifti nibabel.nifti1.Nifti1Image  
    Returns
        ct_coords np.ndarray float (N,3)
    """
    Tmatrix = np.transpose(np.linalg.inv(nifti.affine))
    ct_coords = np.hstack((ct_coords, np.ones((ct_coords.shape[0],1)))) @ Tmatrix
    return ct_coords[:,0:3]
 
def dist(x,y):
    return np.sqrt(np.sum( (x-y)**2))

def euclidean_distance_coords(predictions, ground_truth):
    """
    Given two set of points, find the euclidean distance between them.
    This function is invariant to the order of the points.
    We assume that the points closest to each other are the same points. i.e. represent the same object.
        predictions nd.array nx3
        ground_truth nd.array nx3
    """
    pred_perm = np.zeros_like(predictions)
    dsts = []
    perms = list(itertools.permutations([0, 1, 2, 3]))
    for perm in perms:
        pred_perm[0], pred_perm[1], pred_perm[2], pred_perm[3] = \
        predictions[perm[0]], predictions[perm[1]], predictions[perm[2]], predictions[perm[3]]
        dsts.append(np.mean(np.sqrt( np.sum((pred_perm-ground_truth)**2, axis=1 ))))
    return np.min(dsts) 
    
def create_video(orig_image, dim=0):
    """
    Creates a video given the numpy array along dim
    Args
        orig_image np.ndarray float
        dim int
    Return
        anim matplotlib.animation.FuncAnimation
    """
    fig, ax = plt.subplots()
    plt.close()
    def animator(N): # N is the animation frame number
        if dim == 0:
            ax.imshow(orig_image[N,:,:], cmap='gray') 
        elif dim==1:
            ax.imshow(orig_image[:,N,:], cmap='gray') 
        else:
            ax.imshow(orig_image[:,N,:], cmap='gray') 
            
        ax.axis('off')
        return ax
    PlotFrames = range(0, orig_image.shape[0], 1)
    anim = animation.FuncAnimation(fig, animator,frames=PlotFrames,interval=100)
    rc('animation', html='jshtml') # embed in the HTML for Google Colab
    return anim

def rotate(img, angle):
    """
    Function to rotate an image
    Args
        img np.ndarray float 
        angle float
    Retruns
        dst np.ndarray float
    """
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst 

def procrustes(X, Y):
    """
        Ya = procrustes(X, Y)

    Returns Ya = alpha * (Y - muY) * Q + muX, where muX and muY are the m x n
    matrices whose rows contain copies of the centroids of X and Y, and alpha
    (scalar) and Q (m x m orthogonal matrix) are the solutions to the Procrustes
    + scaling problem

    Inputs: `X` and `Y` are m x n matrices

    Output: `Ya` is an m x n matrix containing the Procrustes-aligned version
    of Y aligned to X and Q the optimal orthogonal matrix

    min_{alpha, Q: Q^T Q = I} |(X - muX) - alpha * (Y - muY) Q|_F
    """
    muX = np.mean(X, axis=0)
    muY = np.mean(Y, axis=0)
    
    X0 = X - muX 
    Y0 = Y - muY 
    # Procrustes rotation
    U, _, V = np.linalg.svd(np.transpose(X0) @ Y0, full_matrices=False)
    V=np.transpose(V)
    Q = V @ np.transpose(U)
    # Optimal scaling
    alpha = np.trace(np.transpose(X0) @ Y0 @ Q) / np.trace(np.transpose(Y0) @ Y0)

    # Align data
    Ya = alpha * (Y0 @ Q) + muX

    return Ya, Q, muX, muY, alpha 

def naive_project(coord_list,points_hull):
    """
    Function to get the coordinates of 2d points on the 3d surface
    It simulates projection of ray along x axis stepwise and uses integers to find intersection
    Args 
        coord_list np.ndarray float
        points_hull np.ndarray float
    Returns
        projected_coord_list np.ndarray int
    """
    coord_list = coord_list[:,1:3]
    coord_list = coord_list.astype(int)
    points_hull = points_hull.astype(int)
    projected_coord_list = []

    for coord in coord_list:
        # find where the last two coordinates are equal
        idx = np.where( ( points_hull[:, 1:3] == np.array(coord)).all(axis=1))
        if len(idx) == 0:
            continue
        else:
            projected_coord_list.append(points_hull[idx][np.argmax(points_hull[idx][:,0])])

    return projected_coord_list

def naive_project2(coord_list,points_hull):
    """
    Function to get the coordinates of 2d points on the 3d surface
    It simulates projection of ray along x axis stepwise and uses integers to find intersection
    Args 
        coord_list np.ndarray float
        points_hull np.ndarray float
    Returns
        projected_coord_list np.ndarray int
    """
    coord_list = coord_list[:,1:3]
    projected_coord_list = []

    for coord in coord_list:
        coord = [100, coord[0], coord[1]]
        ## find the point on the hull closest to cooord
        dist = np.linalg.norm(points_hull - coord, axis=1)
        idx = np.argmin(dist)
        projected_coord_list.append(points_hull[idx])

    return projected_coord_list
def resize_coords(old_coords, old_size, new_size):
    new_coords = []
    Rx = new_size[0]/old_size[0]
    Ry = new_size[1]/old_size[1]
    for i, old_coord in enumerate(old_coords):
        new_coords[i] = [round(Rx*old_coord[0]), round(Ry*old_coord[1])]
    return new_coords

def calc_arc_length(p1, p2, c):
    """
    Calculate the clockwise arc length from p1 to p2 with center c
    """
    x1 = p1[0]; y1 = p1[1]     # Start point
    xc = c[0]; yc = c[1]       # Center point
    r = math.sqrt((x1-xc)**2 + (y1-yc)**2)
    # End point
    x2 = xc+r*(p2[0]-xc)/math.sqrt((p2[0]-xc)**2+(p2[1]-yc)**2)
    y2 = yc+r*(p2[1]-yc)/math.sqrt((p2[1]-yc)**2+(p2[0]-xc)**2)
    x3 = 2*xc-x1; y3 = 2*yc-y1 # Point opposite start point
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    theta = math.acos(1 - (d**2)/(2*r**2))
    if x2 > x3:   # This only works if p1 is bottom point now
        theta = 2*math.pi-theta
    arc_length = r*theta
    return arc_length

def coord_from_arc_length(p1, c, arc_length):
    """
    Calculate the coordinate p2 with arc length from p1
    """
    x1 = p1[0]; y1 = p1[1]     # Start point
    xc = c[0]; yc = c[1]       # Center point
    r = math.sqrt((x1-xc)**2 + (y1-yc)**2)
    circum = 2*math.pi*r
    theta = 2*math.pi*arc_length/circum
    coord = [x1-r*math.sin(theta), y1-r*(1-math.cos(theta))]
    return coord

if __name__ =="__main__":
    import ex
    ex.hey()