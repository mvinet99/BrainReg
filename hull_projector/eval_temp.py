import os
import numpy as np
import scipy.io
import nibabel as nib
import sys
from utils_l import naive_project
from utils_l import math_project
from utils_l import euclidean_distance_coords
from mayavi import mlab
from tqdm import tqdm
import pandas as pd
m1 = []
m2 = []

df = pd.read_csv("split.csv")
df = df[df['split']=='test']

for study_id in tqdm(df.ids[:12]):
    try:
        prect = nib.load(os.path.join("data",
                                      study_id,
                                      "preop_ct.nii"))
        Tmatrix = np.transpose(prect.affine)

        ground_truth_coords = np.load(os.path.join("data",
                                              study_id,
                                              "electrode_locs.npy"))
        temp_coords = np.hstack( (ground_truth_coords, np.ones( (ground_truth_coords.shape[0],1))))
        aligned_coords= (temp_coords @ np.linalg.inv(Tmatrix))[:,0:2]

        hull = scipy.io.loadmat(os.path.join("data",
                                             study_id,
                                             "hull_rh.mat"))
        points_hull = np.array(hull['mask_indices'])

        coords = aligned_coords
        coords = np.hstack( (aligned_coords, np.zeros( (aligned_coords.shape[0],1))))
        coords = np.hstack( (coords, np.ones( (coords.shape[0],1))))
        coords = coords @ Tmatrix
        prediction = np.array(naive_project(coords[:,0:3],points_hull))

        coords = aligned_coords
        coords = np.hstack( (aligned_coords, np.zeros( (aligned_coords.shape[0],1))))
        coords = np.hstack( (coords, np.ones( (coords.shape[0],1))))
        coords = coords @ Tmatrix
        prediction2 = np.array(math_project(coords[:,0:3],points_hull))
        prediction2

        m1.append(euclidean_distance_coords(ground_truth_coords, prediction))
        m2.append(euclidean_distance_coords(ground_truth_coords, prediction2))

        print("MEAN m1")
        print(np.mean(m1))

        print("MEAN m2")
        print(np.mean(m2))
    except:
        pass
print("M1 is")
print(m1)

print("M2 is")
print(m2)

print("MEAN m1")
print(np.mean(m1))
print(np.std(m1))
print("MEAN m2")
print(np.mean(m2))
print(np.std(m2))