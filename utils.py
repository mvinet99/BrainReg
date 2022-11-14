from matplotlib import animation, rc
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

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

def resize_coords(old_coords, old_size, new_size):
    new_coords = []
    Rx = new_size[0]/old_size[0]
    Ry = new_size[1]/old_size[1]
    for i, old_coord in enumerate(old_coords):
        new_coords[i] = [round(Rx*old_coord[0]), round(Ry*old_coord[1])]
    return new_coords

def arc_length(p1, p2, c):
    """
    Function to calculate the arc length between 2 points

    input: 2 points along circle circumference and circle center
    output: arc length in pixels
    """
    x1 = p1[0]; y1 = p1[1]
    x2 = p2[0]; y2 = p2[1]
    xc = c[0]; yc = c[1]
    r = math.sqrt((x1-xc)**2 + (y1-yc)**2)
    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    theta = math.acos(1 - (d**2)/(2*r**2))
    arc_length = r*theta
    return arc_length

if __name__ =="__main__":
    import ex
    ex.hey()