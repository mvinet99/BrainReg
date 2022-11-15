from mayavi import mlab
import os
import imageio
import cv2
import numpy as np
import nibabel as nib
import scipy.io
from fluoro_ct_alignment import project_to_2d
from utils import rotate
from utils import procrustes
from utils import video_to_points


def picker_callback(picker_obj):
    """
    Function to enable picker on a mayavi object
    
    input: picker_obj
    output: print out coordinates of the point picked by 
    a mouse
    
    A mayavi object needs to be defined before running
    this function. Once the object is displayed in the
    mayavi scene, the console will output the coordinates
    of point from the mouse click on the mayavi scene. 
    
    """

    print(picker_obj.get("pick_position"))

SAMPLE_NAME = "DBS_bT20"
HULL_FILE_NAME = os.path.join(SAMPLE_NAME,"hull_rh.mat")
PRECT_FILE_NAME = os.path.join(SAMPLE_NAME,"rpreop_ct.nii")
POSTCT_FILE_NAME = os.path.join(SAMPLE_NAME,"rpostop_ct.nii")
FLUORO_FILE_NAME = os.path.join(SAMPLE_NAME,"fluoro20.tif")
T1_FILE_NAME = os.path.join(SAMPLE_NAME, "T1.nii")

# read .mat File
hull = scipy.io.loadmat(HULL_FILE_NAME)
points_hull=np.array(hull['mask_indices'])

#read pre op CT
prect = nib.load(PRECT_FILE_NAME)
prect_data = np.nan_to_num(np.array(prect.get_fdata()))

#read post op CT
postct = nib.load(POSTCT_FILE_NAME)
postct_data = np.nan_to_num(np.array(postct.get_fdata()))
Tmatrix = np.transpose(prect.affine)

t1 = nib.load(T1_FILE_NAME)
t1_data = np.nan_to_num(np.array(t1.get_fdata()))

# fourth channel is infrared
fluoro = imageio.imread(FLUORO_FILE_NAME)
fluoro = cv2.cvtColor(fluoro, cv2.COLOR_BGR2GRAY)

fig = mlab.figure(bgcolor=(1, 1, 1), size=(1000, 1000))
fig.on_mouse_pick(picker_callback)
source_1 = mlab.pipeline.scalar_field(prect_data)
surface = mlab.pipeline.iso_surface(source_1, 
                          contours=[256,], 
                          opacity=0.5, 
                          colormap = 'black-white')

source_2 = mlab.pipeline.scalar_field(postct_data[:,:128,:])
surface = mlab.pipeline.iso_surface(source_2, 
                          contours=[256,], 
                          opacity=0.8, 
                          colormap = 'black-white')

pins_ct = np.array([[209.0, 147.626220703125, 54.0],
                    [73.0, 73.03683471679688, 52.0],
                    [147.27645874023438, 114.0, 85.0]])

fluoro = rotate(fluoro, 90)
fluoro = cv2.resize(fluoro,(150,150))

shift_down = 330
shift_right = 200

fluoro_new = np.vstack((np.ones((shift_down,)+fluoro.shape[1:], 
                                    dtype=fluoro.dtype), 
                                    fluoro))

fluoro_newnew = np.hstack((np.ones(fluoro_new.shape[:1]+(shift_right,),
                                    dtype=fluoro_new.dtype), 
                                    fluoro_new))

#mlab.imshow(fluoro_newnew)
#mlab.points3d(pins_ct[:,0],pins_ct[:,1],pins_ct[:,2], color = (0.2,1,.2), scale_factor=10)
points_hull = np.hstack( (points_hull, np.ones( (points_hull.shape[0],1))))

points_hull=points_hull@np.linalg.inv(Tmatrix)

mlab.points3d(points_hull[:,0],points_hull[:,1], points_hull[:,2], color = (0.2,1,.2), scale_factor=10)
mlab.show()