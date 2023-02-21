import numpy as np
import nibabel as nib
import numpy as np
from sklearn.cluster import DBSCAN
from utils_l import dist
from ct_localizer import process_study_inference, get_model

def ct_get_landmarks(prect, postct, method="ML"):
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
	if method=="ML":
		ct_ex, factors = process_study_inference(prect)
		model = get_model(width=64, height=64, depth=64)
		model.load_weights("weights/ct_pin_localize.h5")
		prediction = model.predict(np.expand_dims(ct_ex, axis=0))[0]
		factors = 1/np.array(factors)
		prediction[:,0]*=factors[0]
		prediction[:,1]*=factors[1]
		prediction[:,2]*=factors[2]
		prediction *= 60
		pins = prediction
	else:
		pins = get_pins(prect)

	leads = get_lead(postct)
	print('ct_segmentation.py successfully executed.')
	return {"pin": pins, "lead": leads }

def get_pins(nifti_data):

	for idx, layer in enumerate(nifti_data[:,:]):
		if np.all((layer==1)):
			nifti_data[:,:,idx] = np.zeros(nifti_data.shape)

	point_cloud = np.array([0,0,0])		
	for i in range(nifti_data.shape[1]):
		img = nifti_data[:,i,:]
		norm_term = (np.max(img) - np.min(img))
		if(norm_term == 0):
			norm_term = 1.0
		img = (img - np.min(img)) / norm_term
		# get their coordinates and attach i
		white_pixels = img > 0.98
		white_pixel_coords = np.array(np.where(white_pixels==1)).transpose(1,0)
		if len(white_pixel_coords)<4000:
			point_cloud = np.vstack((point_cloud, np.hstack( (white_pixel_coords, np.ones((white_pixel_coords.shape[0] , 1))*i ))))

	model = DBSCAN(eps=25, min_samples=5)
	model.fit_predict(point_cloud)

	point_cloud = point_cloud[model.labels_!=-1]
	labs = model.labels_[model.labels_!=-1]

	a,b = np.unique(labs, return_counts=True)
	sorted_clusters = np.array((a,b)).transpose(1,0)
	sorted_clusters=sorted_clusters[sorted_clusters[:,1].argsort()]
	l1, l2, l3, l4 = sorted_clusters[-1][0], sorted_clusters[-2][0], sorted_clusters[-3][0], sorted_clusters[-4][0], 
	take = (labs==l1) | (labs==l2) |  (labs== l3) |(labs== l4)
	new_pc = point_cloud[take]
	new_labels = labs[take]
	all_labels = np.unique(new_labels)
	min_points = []
	for lab in all_labels:
		cluster = new_pc[new_labels==lab]
		# among all points in the cluster find the point closest to
		midpoint = (128, 128, 88)
		min_dist = np.inf
		min_point = ()
		for p in cluster:
			if dist(p, midpoint) < min_dist:
				min_dist = dist(p,midpoint)
				min_point=p
		min_points.append(min_point)
	min_points = np.array(min_points)
	predictions = np.transpose(np.array([min_points[:,0], min_points[:,2], min_points[:,1]]))
	return predictions

#	pin = np.array([[209.0, 147.626220703125, 54.0],
#				[73.0, 73.03683471679688, 52.0],
#				[77.19986724853516, 71.0, 148.0],
#				[203.0, 149.6582489013672, 153.0]])
def get_lead(nifti_data):

	point_cloud = np.array([0,0,0])
	for i in range(nifti_data.shape[1]):
		img = nifti_data[:,i,:]
		norm_term = (np.max(img) - np.min(img))
		if(norm_term == 0):
			norm_term = 1.0
		img = (img - np.min(img)) / norm_term
		# get their coordinates and attach i
		white_pixels = (img==1.0)
		white_pixel_coords = np.array(np.where(white_pixels == 1)).transpose(1,0)
		if len(white_pixel_coords)==1:
			point_cloud = np.vstack((point_cloud, np.hstack( (white_pixel_coords, np.ones((white_pixel_coords.shape[0] , 1))*i )))) 
	model = DBSCAN(eps=25, min_samples=10)
	model.fit_predict(point_cloud)

	point_cloud = point_cloud[model.labels_!=-1]
	labs = model.labels_[model.labels_!=-1]
 
	mid = np.array([128, 88, 128])
	smallest_dist=np.inf
	best_point=None
	best_point_id=-1
	for idx, point in enumerate(point_cloud):
		if dist(point, mid)< smallest_dist:        
			smallest_dist = dist(point,mid)
			best_point=point
			best_point_id= idx
	lead1 = best_point
	# delete a cluster from which this point is
	new_pc = point_cloud[labs != labs[best_point_id]]

	smallest_dist=np.inf
	best_point=None
	for point in new_pc:
		if dist(point, mid)< smallest_dist:        
			smallest_dist = dist(point,mid)
			best_point=point

	lead2 = best_point

	preds = np.stack((lead1,lead2))

	return np.transpose(np.array([preds[:,0], preds[:,2], preds[:,1]]))

if __name__ == "__main__":
	from utils_l import from_hull_to_ct_coords
	from utils_l import euclidean_distance_coords
	import os
	import pandas as pd
	from tqdm import tqdm

	df = pd.read_csv("split.csv")
	df = df.loc[df['split'] == 'test']
	TEST_SAMPLES = np.array(df.ids)
	# working great on
#	TEST_SAMPLES = ["DBS_bG67",
#					"DBS_bG66",
#					"DBS_bG64",
#					"DBS_bG56",
#					"DBS_bG30",
#					"DBS_bG30",
#					"DBS_bG28",
#					"DBS_bG22",
#					"DBS_bG09",
#					"DBS_bG06"]

	acc=[]                  
	for SAMPLE_NAME in tqdm(TEST_SAMPLES):
		if os.path.exists(os.path.join("data", SAMPLE_NAME, "preop_ct.nii")):
			GT = np.load(os.path.join("data/", SAMPLE_NAME,"pin_tips.npy"))
			GT = from_hull_to_ct_coords(GT, nib.load(os.path.join("data/", SAMPLE_NAME,"preop_ct.nii")))

			prect = nib.load("data/"+SAMPLE_NAME+"/preop_ct.nii")
			prect_data = np.nan_to_num(np.array(prect.get_fdata()))

#			prect=prect_data
#			ct_ex, factors = process_study_inference(prect)
#			model = get_model(width=64, height=64, depth=64)
#			model.load_weights("weights/ct_pin_localize.h5")
#			prediction = model.predict(np.expand_dims(ct_ex, axis=0))[0]
#			factors = 1/np.array(factors)
#			prediction[:,0]*=factors[0]
#			prediction[:,1]*=factors[1]
#			prediction[:,2]*=factors[2]
#			prediction *= 60
#			res = prediction

			res = get_pins(prect_data)
			acc.append(euclidean_distance_coords(res, GT))
			print(acc)
	print(np.mean(acc))
	print(f"{np.mean(acc)} +=- {np.std(acc)}")
