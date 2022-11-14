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

SAMPLE_NAME = "data/DBS_bG02"
NIFTI_PATH = os.path.join(SAMPLE_NAME,"coregistered_preopCT_DBS_bG02.nii")

nifti = nib.load(NIFTI_PATH)
nifti_data = np.nan_to_num(np.array(nifti.get_fdata()))

source = mlab.pipeline.scalar_field(nifti_data)
surface = mlab.pipeline.iso_surface(source,
                          contours=[256,], 
                          opacity=0.8, 
                          colormap = 'black-white')
mlab.show()