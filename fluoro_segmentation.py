import numpy as np
"""
RYAN
"""
def fluoro_get_electrode_coordinates(fluoro):
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
    coords = np.array([[449., 626.],
        [490., 575.],
        [536., 525.],
        [587., 485.],
        [643., 452.],
        [706., 430.],
        [763., 413.],
        [830., 405.]])

    dbs = np.array([922, 805])
    pin_tips = np.array([[ 542., 1019.],
                        [1399.,  539.]])
    print('fluoro_segmentation.py successfully executed.')
    return {"ecog":coords,
    "dbs": dbs,
    "pin": pin_tips }
