from fluoro_segmentation import fluoro_get_electrode_coordinates
from ct_segmentation import ct_get_landmarks 
from fluoro_ct_alignment import project_to_2d
from hull_projection import project_to_3d

def run_pipeline(fluoro, prect, postct, hull, Tmatrix):
	dic = fluoro_get_electrode_coordinates(fluoro)
	dic2 = ct_get_landmarks(prect, postct)

