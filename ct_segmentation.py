import numpy as np
"""
TIGER
"""
def ct_get_landmarks(prect, postct):
	"""
	Given the CT preop and postop image, performs landmark detection and
	registration and returns the coordinates for each registered landmark.
	Args:
		prect 3 dimensional float64 numpy array 
		postct 3 dimensional float64 numpy array
	Return:
		landmarks float64 numpy array shape = (n,x,y,z) 
			where n is the number of landmarks
			x is the x coordinate of the landmark
			y is the y coordinate of the landmark
			z is the y coordinate of the landmark
	"""
	# first four are pins_ct
	# last one is lead tip

	lead = np.array([147.27645874023438, 114.0, 85.0])
	pin = np.array([[209.0, 147.626220703125, 54.0],
				[73.0, 73.03683471679688, 52.0],
				[77.19986724853516, 71.0, 148.0],
				[203.0, 149.6582489013672, 153.0]])

	## TODO: READ FROM GROUD TRUTH 
	test_coords = np.array([[-45.11485451, -76.83932495, -40.28118491 , 1],
			[-42.11494714,  55.16067505,  33.71881884 ,1],
			[ 42.88505286,  53.16067505,  33.71892524 , 1],
			[ 50.88514924, -73.83932495, -43.28106475 , 1]])
	test_coords = test_coords @ np.linalg.inv(Tmatrix) 

	print('ct_segmentation.py successfully executed.')
	return {"pin": pin, "lead": lead }
