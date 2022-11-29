import numpy as np
import nibabel as nib
import numpy as np
from sklearn.cluster import DBSCAN
from utils import dist

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
	pins = get_pins(prect)
	leads = get_lead(postct)
	#lead = np.array([147.27645874023438, 114.0, 85.0])
	print('ct_segmentation.py successfully executed.')
	return {"pin": pins, "lead": leads }

def get_pins(nifti_data):

	for idx, layer in enumerate(nifti_data[:,:]):
		if np.all((layer==1)):
			nifti_data[:,:,idx] = np.zeros(nifti_data.shape)

	point_cloud = np.array([0,0,0])		
	for i in range(nifti_data.shape[1]):
		img = nifti_data[:,i,:]
		img = (img - np.min(img)) / (np.max(img) - np.min(img))
		# get their coordinates and attach i
		white_pixels = img > 0.98
		white_pixel_coords = np.array(np.where(white_pixels==1)).transpose(1,0)
		if len(white_pixel_coords)<4000:
			point_cloud = np.vstack((point_cloud, np.hstack( (white_pixel_coords, np.ones((white_pixel_coords.shape[0] , 1))*i ))))

	model = DBSCAN(eps=25, min_samples=5)
	model.fit_predict(point_cloud)
	pred = model.fit_predict(point_cloud)
	a,b = np.unique(model.labels_, return_counts=True)
	sorted_clusters = np.array((a,b)).transpose(1,0)
	sorted_clusters=sorted_clusters[sorted_clusters[:,1].argsort()]
	l1, l2, l3, l4 = sorted_clusters[-1][0], sorted_clusters[-2][0], sorted_clusters[-3][0], sorted_clusters[-4][0], 
	take = (model.labels_==l1) | (model.labels_==l2) |  (model.labels_== l3) |(model.labels_== l4)
	new_pc = point_cloud[take]
	new_labels = model.labels_[take]
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
		img = (img - np.min(img)) / (np.max(img) - np.min(img))
		# get their coordinates and attach i
		white_pixels = (img==1.0)
		white_pixel_coords = np.array(np.where(white_pixels == 1)).transpose(1,0)
		if len(white_pixel_coords)==1:
			point_cloud = np.vstack((point_cloud, np.hstack( (white_pixel_coords, np.ones((white_pixel_coords.shape[0] , 1))*i )))) 
	model = DBSCAN(eps=25, min_samples=10)
	model.fit_predict(point_cloud)
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
	new_pc = point_cloud[model.labels_ != model.labels_[best_point_id]]

	smallest_dist=np.inf
	best_point=None
	for point in new_pc:
		if dist(point, mid)< smallest_dist:        
			smallest_dist = dist(point,mid)
			best_point=point

	lead2 = best_point
	return np.stack((lead1,lead2))

if __name__ == "__main__":
	from utils import from_hull_to_ct_coords
	from utils import euclidean_distance_coords
	import os
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
		GT = np.load(os.path.join("data/", SAMPLE_NAME,"pin_tips.npy"))
		GT = from_hull_to_ct_coords(GT, nib.load(os.path.join("data/", SAMPLE_NAME,"preop_ct.nii")))

		prect = nib.load("data/"+SAMPLE_NAME+"/preop_ct.nii")
		prect_data = np.nan_to_num(np.array(prect.get_fdata()))

		res = get_pins(prect_data)
		acc.append(euclidean_distance_coords(res, GT))
	print(np.mean(acc))