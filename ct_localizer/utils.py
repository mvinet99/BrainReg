import numpy as np
import tensorflow as tf
from tensorflow import keras

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