import tensorflow as tf
import numpy as np

@tf.custom_gradient
def spike_function(v_to_threshold):
    z = tf.cast(tf.greater(v_to_threshold, 1.), tf.float32)

    def grad(dy):
        return [dy * tf.maximum(1 - tf.abs(v_to_threshold - 1), 0)]
        # @negative: v_to_threshold < 0 -> dy*0
        # @rest: v_to_threshold = 0 -> dy*0+
        # @thresh: v_to_threshold = 1 -> dy*1
        # @+thresh: v_to_threshold > 1 -> dy*1-
        # @2thresh: v_to_threshold > 2 -> dy*0

    return z, grad

def reset_by_threshold(old_z, decay, old_v, threshold):
    return old_z * threshold

class LiNeuronCell(tf.keras.layers.Layer):
    def __init__(self, n_neurons, tau=20., **kwargs):
        super(LiNeuronCell, self).__init__(**kwargs)
        self.n_neurons = n_neurons
        self.tau = tau

    def get_config(self):
        config = super().get_config()
        config.update({'n_neurons': self.n_neurons,
                       'tau': self.tau})
        return config

    def build(self, input_shape):
        self.n_in = input_shape[-1]

        self.decay = tf.exp(-1/self.tau)

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True, name='w_in')

        self.info_n_neurons = self.n_neurons
        self.info_n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        self.info_n_synapses = self.w_in.numpy().size

    @property
    def state_size(self):
        return self.n_neurons

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        del inputs  # Unused

        zeros = tf.zeros((batch_size, self.n_neurons), dtype=dtype)
        return zeros

    def call(self, input_at_t, states_at_t):
        old_v = states_at_t[0]

        dim = tf.reduce_prod(tf.shape(input_at_t)[1:])
        new_input = tf.reshape(input_at_t, [-1, dim])

        i_t = tf.matmul(new_input, self.w_in)
        
        new_v = old_v + i_t
        new_z = tf.nn.softmax(new_v)

        return (new_z, new_v), new_v


class LifNeuronCell(tf.keras.layers.Layer):
    def __init__(self, n_neurons, decay_train_mode=0, threshold_train_mode=0, tau=20., initial_threshold=0.1, activation_function=spike_function, reset_function=reset_by_threshold, **kwargs):
        super(LifNeuronCell, self).__init__(**kwargs)
        self.n_neurons = n_neurons

        # 0 = no training, 1 = single value per layer, 2 = vector
        self.decay_train_mode = decay_train_mode
        self.threshold_train_mode = threshold_train_mode
        self.reset_function = reset_function
        self.tau = tau
        self.initial_threshold = initial_threshold

        self.activation_function = activation_function

    def get_config(self):
        config = super().get_config()
        config.update({'n_neurons': self.n_neurons,
                       'decay_train_mode': self.decay_train_mode,
                       'threshold_train_mode': self.threshold_train_mode,
                       'reset_function': self.reset_function,
                       'tau': self.tau,
                       'initial_threshold': self.initial_threshold,
                       'activation_function': self.activation_function})
        return config

    def build(self, input_shape):
        self.n_in = input_shape[-1]

        decay = tf.cast(tf.exp(-1 / self.tau), dtype=self.dtype)
        if self.decay_train_mode == 0:
            self.decay = tf.Variable(initial_value=decay, trainable=False, dtype=self.dtype, name='decay')
        elif self.decay_train_mode == 1:
            self.decay = tf.Variable(initial_value=decay, trainable=True, dtype=self.dtype, name='decay')
        elif self.decay_train_mode == 2:
            self.decay = tf.Variable(initial_value=decay * tf.ones((self.n_neurons)), trainable=True, dtype=self.dtype,
                                     name='decay')
        else:
            print('Wrong decay train mode specified')

        if self.threshold_train_mode == 0:
            self.threshold = tf.Variable(initial_value=self.initial_threshold, trainable=False, dtype=self.dtype, name='threshold')
        elif self.threshold_train_mode == 1:
            self.threshold = tf.Variable(initial_value=self.initial_threshold, trainable=True, dtype=self.dtype, name='threshold')
        elif self.threshold_train_mode == 2:
            self.threshold = tf.Variable(initial_value=self.initial_threshold * tf.ones((self.n_neurons)), trainable=True, dtype=self.dtype,
                                     name='threshold')
        else:
            print('Wrong threshold train mode specified')

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True)

        self.info_n_neurons = self.n_neurons
        self.info_n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        self.info_n_synapses = self.w_in.numpy().size

    @property
    def state_size(self):
        return (self.n_neurons, self.n_neurons)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        del inputs  # Unused

        zeros = tf.zeros((batch_size, self.n_neurons), dtype=dtype)
        return (zeros, zeros)

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        dim = tf.reduce_prod(tf.shape(input_at_t)[1:])
        new_input = tf.reshape(input_at_t, [-1, dim])
        i_t = tf.matmul(new_input, self.w_in)
        i_reset = self.reset_function(old_z, self.decay, old_v, self.threshold)
        
        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v/self.threshold)

        return (new_z, new_v), (new_v, new_z)



class RecurrentLifNeuronCell(LifNeuronCell):
    def build(self, input_shape):
        self.n_in = input_shape[-1]

        w_in = tf.random.normal((self.n_in, self.n_neurons), dtype=self.dtype)
        self.w_in = tf.Variable(initial_value=w_in / np.sqrt(self.n_in), trainable=True, name='w_in')

        w_rec = tf.random.normal((self.n_neurons, self.n_neurons), dtype=self.dtype)
        w_rec = tf.linalg.set_diag(w_rec, np.zeros(self.n_neurons))
        self.w_rec = tf.Variable(initial_value=w_rec / np.sqrt(self.n_neurons), trainable=True, name='w_rec')

        decay = tf.cast(tf.exp(-1/self.tau), dtype=self.dtype)
        if self.decay_train_mode == 0:
            self.decay = tf.Variable(initial_value=decay, trainable=False, dtype=self.dtype, name='decay')
        elif self.decay_train_mode == 1:
            self.decay = tf.Variable(initial_value=decay, trainable=True, dtype=self.dtype, name='decay')
        elif self.decay_train_mode == 2:
            self.decay = tf.Variable(initial_value=decay*tf.ones((self.n_neurons)), trainable=True, dtype=self.dtype, name='decay')
        else:
            print('Wrong decay train mode specified')

        if self.threshold_train_mode == 0:
            self.threshold = tf.Variable(initial_value=self.initial_threshold, trainable=False, dtype=self.dtype, name='threshold')
        elif self.threshold_train_mode == 1:
            self.threshold = tf.Variable(initial_value=self.initial_threshold, trainable=True, dtype=self.dtype, name='threshold')
        elif self.threshold_train_mode == 2:
            self.threshold = tf.Variable(initial_value=self.initial_threshold * tf.ones((self.n_neurons)), trainable=True, dtype=self.dtype,
                                     name='threshold')
        else:
            print('Wrong threshold train mode specified')

        self.info_n_neurons = self.n_neurons
        self.info_n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in self.trainable_variables])
        self.info_n_synapses = self.w_in.numpy().size + self.w_rec.numpy().size

    def call(self, input_at_t, states_at_t):
        old_v, old_z = states_at_t

        dim = tf.reduce_prod(tf.shape(input_at_t)[1:])
        new_input = tf.reshape(input_at_t, [-1, dim])

        i_t = tf.matmul(new_input, self.w_in) + tf.matmul(old_z, self.w_rec)
        i_reset = self.reset_function(old_z, self.decay, old_v, self.threshold)

        new_v = self.decay * old_v + (1.0 - self.decay) * i_t - i_reset
        new_z = self.activation_function(new_v/self.threshold)

        return (new_z, new_v), (new_v, new_z)


