import tensorflow as tf
import numpy as np

def split_to_string(split, prefix='p'):
    split_str = ''
    for i in split:
        split_str += prefix + str(i) + '+'
    return split_str[:-1]

def loo_split(loo_i, max_i):
    eval_split_i = [loo_i]
    train_split_i = np.setdiff1d(np.arange(0, max_i), eval_split_i)
    
    eval_split = split_to_string(eval_split_i)
    train_split = split_to_string(train_split_i)
    return train_split, eval_split

def loo_split_aug(loo_i, max_i, aug):
    eval_split_i = [loo_i]
    train_split_i = np.setdiff1d(np.arange(0, max_i), eval_split_i)
    
    eval_split = split_to_string(eval_split_i, '0_p')

    train_split = ''
    for i in aug:
        train_split += split_to_string(train_split_i, str(i) + '_p') + '+'
    return train_split[:-1], eval_split

class Mapping():
    def __init__(self, max_length) -> None:
        self.max_length = max_length
        self.repeat_factor = 1

    def pad(self, x_raw, label):
        #x_raw = binarize_tf(x_raw, 0.02)
        length = tf.shape(x_raw)[0]
        indices = tf.cast(tf.linspace(0.0, tf.cast(length-1, dtype=tf.float32), self.max_length), dtype=tf.int64)
        
        x = tf.gather(x_raw, indices, axis=0)
        x = x - x[0]
        y = label   
        return x, y

    def pad_sequence(self, x_raw, label):
        #x_raw = binarize_tf(x_raw, 0.02)
        length = tf.shape(x_raw)[0]
        indices = tf.cast(tf.linspace(0.0, tf.cast(length-1, dtype=tf.float32), self.max_length), dtype=tf.int64)
        
        x = tf.gather(x_raw, indices, axis=0)
        y = tf.repeat(tf.expand_dims(label, axis=-1), self.max_length, axis=-1)
        return x, y

    def binarize_tf(self, data, threshold=0.02):
        t_length = tf.shape(data)[0]
        flattened = tf.reshape(data, (t_length, -1))
        
        rdm_length = tf.cast(tf.shape(flattened)[1], dtype=tf.float32)
        threshold_index = tf.cast(rdm_length - rdm_length * threshold, dtype=tf.int32)
        top_values = tf.argsort(tf.argsort(flattened)) >= threshold_index
        nonzero = flattened != 0
        top_values_nonzero = tf.cast(tf.logical_and(top_values, nonzero), dtype=tf.int32)
        
        return tf.reshape(top_values_nonzero, tf.shape(data))
        
    def pad_zeros(self, x_raw, label):
        length = tf.shape(x_raw)[0]
        x_raw = x_raw[:tf.math.minimum(length, self.max_length)]
        length = tf.shape(x_raw)[0]
        paddings = [[0, self.max_length-length], [0, 0], [0, 0], [0, 0]]
        
        x = tf.pad(x_raw, paddings, 'CONSTANT')
        #y = tf.repeat(tf.expand_dims(label, axis=-1), length, axis=-1)
        #y = tf.pad(y, [[0, max_length-length]])
        y = label
        
        return x, y

    def repeat(self, x_raw, label):
        x = tf.repeat(x_raw, self.repeat_factor, axis=0)

        return x, label

    def pad_keras(self, x_raw, label):
        x = tf.keras.preprocessing.sequence.pad_sequences(x_raw, maxlen=30)
        y = tf.repeat(tf.expand_dims(label, axis=-1), 30, axis=-1)
        
        return x, y