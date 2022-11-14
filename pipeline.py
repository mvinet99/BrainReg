from fluoro_segmentation import fluoro_get_coordinates
from ct_segmentation import ct_get_landmarks 
from fluoro_ct_alignment import project_to_2d
from hull_projection import project_to_3d
import numpy as np

def run_pipeline(fluoro, prect, postct, hull, Tmatrix):
	fluoro_dic = fluoro_get_coordinates(fluoro)
	ct_dic = ct_get_landmarks(prect, postct)
	aligned_coords = project_to_2d(postct, fluoro, fluoro_dic["pin"], ct_dic["pin"], fluoro_dic["ecog"])
	predictions = project_to_3d(np.array(aligned_coords), hull, Tmatrix)
	return predictions
