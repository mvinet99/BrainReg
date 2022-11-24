from localizer import pin_localizer,accuracy, from_hull_to_ct_coords
import os
import numpy as np
import nibabel as nib

SAMPLE_NAME = "DBS_bT20"
TEST_SAMPLES = ["DBS_bG67",
                "DBS_bG66",
                "DBS_bG64",
				"DBS_bG56",
				"DBS_bG30",
				"DBS_bG30",
				"DBS_bG28",
				"DBS_bG22",
				"DBS_bG09",
				"DBS_bG06"]
acc=[]                  
for SAMPLE_NAME in TEST_SAMPLES:
	GT = np.load(os.path.join("../data/", SAMPLE_NAME,"pin_tips.npy"))
	GT = from_hull_to_ct_coords(GT, nib.load(os.path.join("../data/", SAMPLE_NAME,"preop_ct.nii")))
	predictions = pin_localizer(f"../data/{SAMPLE_NAME}")
	acc.append(accuracy(predictions, GT))
print(np.mean(acc))
#DBS_bG67
#DBS_bG66
#DBS_bG64
#DBS_bG56
#DBS_bG30
#DBS_bG30
#DBS_bG28
#DBS_bG22
#DBS_bG09
#DBS_bG06
# Mean Euclidan distance 3.3485 