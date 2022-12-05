import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
import os
import nibabel as nib


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 64
    desired_height = 64
    
    # Compute depth factor
    f0 = desired_width /  img.shape[0]
    f1 = desired_height /  img.shape[1]
    f2 = desired_depth /   img.shape[2]

    img = ndimage.zoom(img, (f0, f1, f2), order=1)
    return img, f0, f1,f2

def process_study_inference(prect):
    """Read and resize volume"""
    
    # Normalize
    volume = normalize(prect)
    # Resize width, height and depth
    volume, f0, f1, f2 = resize_volume(volume)
     
    return volume, [f0,f1,f2]
def process_study(path):
    """Read and resize volume"""
    
    ct_path = os.path.join(path, "preop_ct.nii")
    pins_path =  os.path.join(path, "pin_tips.npy")
    
    nifti = nib.load(ct_path)
    nifti_data = np.nan_to_num(np.array(nifti.get_fdata()))
    coords = np.load(pins_path)
    coords = from_hull_to_ct_coords(coords, nifti).squeeze()    
    # Normalize
    volume = normalize(nifti_data)
    # Resize width, height and depth
    volume, f0, f1, f2 = resize_volume(volume)
    
    coords[:,0]*=f0
    coords[:,1]*=f1
    coords[:,2]*=f2
    
    coords = sort_points_by_distance_from_origin(coords)
    
    return volume, coords, [f0,f1,f2]

def sort_points_by_distance_from_origin(points):
    """
        Args:
            points 4x3
    """
    keys = np.argsort(np.linalg.norm(points,axis=1))
    return points[keys]

def euc_dist_tf(y_true, y_pred):
    """
        Args:
            y_true tf tensor (batch_size, 4,3) 
            y_pred tf tensor (batch_size, 4,3)
    """
    return tf.math.reduce_mean(tf.math.reduce_euclidean_norm(y_true-y_pred, axis=-1))

def get_four_max_points_tensor(array):
    """
    Function to get four max points of a 3d tensor
    Args
        array tf.Tensor float
    Returns
        four_max_points tf.Tensor float
    """
    return tf.transpose(tf.unravel_index(tf.nn.top_k(tf.reshape(array, [-1]),4).indices,[64,64,64]),[1,0])

class EuclideanDistance(keras.metrics.Metric):
    
    def __init__(self, name="ed", **kwargs):
        super().__init__(name=name, **kwargs)
        
        self.ed_sum = self.add_weight(name="ed_sum",
                                      initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros",
            dtype="int32")
        
    def update_state(self, y_true, y_pred, sample_weight=None):

        ed = euc_dist_tf(y_true, y_pred)
        self.ed_sum.assign_add(ed)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)
        
    def result(self):
        return self.ed_sum / tf.cast(self.total_samples, tf.float32)
    def reset_state(self):
        self.ed_sum.assign(0.)
        self.total_samples.assign(0)
def from_hull_to_ct_coords(ct_coords, nifti):
    """
    Converts the coordinates from the hull to the coordinates of the CT scan
    Args:
        ct_coords np.ndarray float (N,3)
        nifti nibabel.nifti1.Nifti1Image  
    Returns
        ct_coords np.ndarray float (N,3)
    """
    Tmatrix = np.transpose(np.linalg.inv(nifti.affine))
    ct_coords = np.hstack((ct_coords, np.ones((ct_coords.shape[0],1)))) @ Tmatrix
    return ct_coords[:,0:3]