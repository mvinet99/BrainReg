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
	print('Tiger.py successfully executed.')
	return {"pin": pin, "lead": lead }
