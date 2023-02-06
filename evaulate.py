# Import dependencies

from ct_segmentation import ct_get_landmarks
from hull_projection import project_to_3d
from fluoro_segmentation import fluoro_get_coordinates
import fluoro_ct_alignment
from fluoro_ct_alignment import project_to_2d
import nibabel as nib
import numpy as np
import cv2
import scipy.io
import os
import imageio
import math
import circle_fit as cf
from utils import euclidean_distance_coords
from tabulate import tabulate
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("split.csv")
df = df[df['split'] == 'train']

acc = []
cnt=len(df.ids)
for SAMPLE_NAME in  tqdm(df.ids):
	try:
		# LOAD THE DATA
		HULL_FILE_NAME = os.path.join('data/' + SAMPLE_NAME,"hull_rh.mat")
		PRECT_FILE_NAME = os.path.join('data/' + SAMPLE_NAME,"preop_ct.nii")
		POSTCT_FILE_NAME = os.path.join('data/' + SAMPLE_NAME,"postop_ct.nii")
		FLUORO_FILE_NAME = os.path.join('data/' + SAMPLE_NAME,"fluoro.tif")
		GT_NAME = os.path.join('data/' + SAMPLE_NAME,"electrode_locs.npy")

		# read .mat File
		hull = scipy.io.loadmat(HULL_FILE_NAME)
		points_hull=np.array(hull['mask_indices'])

		#read pre op CT
		prect = nib.load(PRECT_FILE_NAME)
		prect_data = np.nan_to_num(np.array(prect.get_fdata()))
		Tmatrix = np.transpose(prect.affine)

		#read post op CT
		postct = nib.load(POSTCT_FILE_NAME)
		postct_data = np.nan_to_num(np.array(postct.get_fdata()))

		# fourth channel is infrared
		fluoro = imageio.imread(FLUORO_FILE_NAME)
		#fluoro = cv2.cvtColor(fluoro, cv2.COLOR_BGR2GRAY)
		ground_truth = np.load(GT_NAME) 

		#Load groud truth 2D aligned electrode coordinates
		new_T = np.hstack((ground_truth, np.ones((ground_truth.shape[0],1))))
		Transformed = new_T @ np.linalg.inv(Tmatrix)
		aligned_coords_gt = np.delete(Transformed, obj=2, axis=1)
		aligned_coords_gt = np.delete(aligned_coords_gt, obj=2, axis=1)
   
		# SEGMENT FROM FLUORO
		# Run first component script
		coords_dict = fluoro_get_coordinates(fluoro)
		coords_dict['ecog'] = np.array(coords_dict['ecog'])
		coords_dict['pin'] = coords_dict['pin'][0].reshape(1,-1)
		coords_dict['dbs']= np.array(coords_dict['dbs'])

  
		# CT SEGMENTATION
		coords3d_dic = ct_get_landmarks(prect_data, postct_data)
		pins_ct = coords3d_dic['pin']

  
		# ALIGNMENT
		fluoro2 = cv2.cvtColor(fluoro, cv2.COLOR_BGR2GRAY)
		dbs = np.array([coords_dict['dbs']])
		pins2d = coords_dict['pin']
		pins_fl = np.vstack([pins2d, dbs.squeeze()])
		coords_2d = coords_dict['ecog']
		pins3d = coords3d_dic['pin']
		lead_ct = coords3d_dic['lead']
		pins_ct = np.concatenate((pins3d,lead_ct),axis=0)
		aligned_coords = project_to_2d(postct_data,fluoro2,pins_fl,pins_ct,coords_2d, aligned_coords_gt)

		# PROJECT
		predictions = project_to_3d(aligned_coords, points_hull, Tmatrix)
		predictions = np.array(predictions)
		predictions 

		ground_truth = ground_truth[:len(predictions)]

		acc.append(euclidean_distance_coords(predictions, ground_truth))
 
	#except:
	except Exception as e:
		print(e)
		cnt-=1
		print(f"DOESNT WORK FOR {SAMPLE_NAME}")
	
 
print("ACCURACY IS {}".format(np.mean(acc)))
print("STD IS {}".format(np.std(acc)))
print(f"RAN ON FOR {cnt}")