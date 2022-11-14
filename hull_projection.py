import numpy as np
from mayavi import mlab
from utils import naive_project
""" 
MILOS 
"""
def project_to_3d(aligned_coords, points_hull, Tmatrix):
	"""
	Args:
		aligned_coords  float64 numpy array shape = (n,x,y)  
		points_hull numpy array (n,3) where n are nodes of the hull
	Return:
		prediction float64 numpy array shape=(n,x,y,z)
	"""
	## I hardcode this for now until output from CT alignment is correct
	aligned_coords = np.array([[203, 83.00006068],
						[198,  76.00005692],
						[192,  70.00005567],
						[185,  64.00005567],   
						[177, 60.00005692],
						[169, 57.00005317],
						[161,  54.00004941],
						[151, 53.00005442]])
	coords=aligned_coords
	coords = np.hstack( (aligned_coords, np.zeros( (aligned_coords.shape[0],1))))
	coords = np.hstack( (coords, np.ones( (coords.shape[0],1))))
	coords = coords @ Tmatrix
	prediction = naive_project(coords[:,0:3],points_hull)
	print("hull_projection.py executed succesfully")
	print(prediction)
	return prediction
# Expected result:
#np.array([[ 48.88512421, -73.83932495,  21.71887207],
#         			 [ 45.88512421, -68.83932495,  28.71887207],
#        			 [ 44.88512421, -62.83932495,  34.71887207],
#         			 [ 44.88512421, -55.83932495,  40.71887207],
#         			 [ 45.88512421, -47.83932495,  44.71887207],
#         			 [ 42.88512421, -39.83932495,  47.71887207],
#        			 [ 39.88512421, -31.83932495,  50.71887207],
#        			 [ 43.88512421, -21.83932495,  51.71887207]])
if __name__ == "__main__":
	aligned_coords = np.array([[203, 83.00006068],
					[198,  76.00005692],
					[192,  70.00005567],
					[185,  64.00005567],   
					[177, 60.00005692],
					[169, 57.00005317],
					[161,  54.00004941],
					[151, 53.00005442]])
	import scipy.io
	import nibabel as nib
	hull = scipy.io.loadmat("DBS_bT20/hull_rh.mat")
	points_hull=np.array(hull['mask_indices'])	

	prect = nib.load("DBS_bt20/rpreop_ct.nii")
	Tmatrix = np.transpose(prect.affine)

	print(project_to_3d(aligned_coords, points_hull, Tmatrix))
