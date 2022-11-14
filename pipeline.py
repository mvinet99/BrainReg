from fluoro_segmentation import fluoro_get_coordinates
from ct_segmentation import ct_get_landmarks 
from fluoro_ct_alignment import project_to_2d
from hull_projection import project_to_3d
import numpy as np

def run_pipeline(fluoro, prect, postct, hull, Tmatrix):
	fluoro_dic = fluoro_get_coordinates(fluoro)
	print(fluoro_dic)
	#ct_dic = ct_get_landmarks(prect, postct)
	#aligned_coords = project_to_2d(postct, fluoro, fluoro_dic["pin"], ct_dic["pin"], fluoro_dic["ecog"])
	pins_ct = np.array([[209, 147.6262207, 54],[73, 73.0368, 52], [77.19986725, 71, 148], [203, 149.658, 153]])
	pins_fl = np.array([[922, 805],[542, 1019], [1399,539]])
	coords_2d = np.array([[449,626],[490,575],[536,525],[587,485],[643,452],[706,430],[763,413],[830,405]])
	aligned_coords = project_to_2d(postct, fluoro, pins_fl, pins_ct, coords_2d)
	predictions = project_to_3d(np.array(aligned_coords), hull, Tmatrix)
	return predictions
