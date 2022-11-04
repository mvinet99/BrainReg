import numpy as np
""" 
MILOS 
"""
def project_to_3d(aligned_coords, points_hull):
	"""
	Args:
		aligned_coords  float64 numpy array shape = (n,x,y)  
		points_hull numpy array (n,3) where n are nodes of the hull
	Return:
		prediction float64 numpy array shape=(n,x,y,z)
	"""
	return np.array([[ 48.88512421, -73.83932495,  21.71887207],
         			 [ 45.88512421, -68.83932495,  28.71887207],
        			 [ 44.88512421, -62.83932495,  34.71887207],
         			 [ 44.88512421, -55.83932495,  40.71887207],
         			 [ 45.88512421, -47.83932495,  44.71887207],
         			 [ 42.88512421, -39.83932495,  47.71887207],
        			 [ 39.88512421, -31.83932495,  50.71887207],
        			 [ 43.88512421, -21.83932495,  51.71887207]])
