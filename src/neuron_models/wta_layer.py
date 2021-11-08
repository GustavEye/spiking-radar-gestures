import tensorflow as tf

class WTA(tf.keras.layers.Layer):   
    def __init__(self, threshold=0.05, **kwargs):
        super(WTA, self).__init__(**kwargs)
        self.threshold = threshold

    def get_config(self):
        config = super().get_config()
        config.update({'threshold': self.threshold})
        return config
        
    def call(self, inputs):
        data = inputs
        shape = tf.shape(data)
        flat_length = shape[2]*shape[3]
        #[batch, time, vel, range, channel]
        flattened = tf.reshape(data, [shape[0], shape[1], shape[2]*shape[3], shape[4]])

        rdm_length = tf.cast(flat_length, dtype=tf.float32)
        threshold_index = tf.cast(rdm_length - rdm_length * self.threshold, dtype=tf.int32)
        top_values = tf.argsort(tf.argsort(flattened, axis=-2), axis=-2) >= threshold_index
        #nonzero = flattened != 0
        top_values_nonzero = tf.cast(tf.logical_and(top_values, tf.math.not_equal(flattened, 0)), dtype=tf.float32)

        return tf.reshape(top_values_nonzero, shape)

class WTA_mean(tf.keras.layers.Layer):   
        
    def call(self, inputs):
        data = inputs
        shape = tf.shape(data)
        #[batch, time, vel, range, channel]
        flattened = tf.reshape(data, [shape[0], shape[1], shape[2]*shape[3], shape[4]])

        rdm_mean = tf.reduce_mean(flattened, axis=-2, keepdims=True)
        top_values = tf.where(tf.greater(flattened, rdm_mean), 1.0, 0.0)

        return tf.reshape(top_values, shape)