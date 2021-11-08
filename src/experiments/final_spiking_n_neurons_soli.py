import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from src.neuron_models import wta_layer, lif_neuron

HPARAMS = [
    hp.HParam('pool_size', hp.Discrete([1])),
    hp.HParam('wta_threshold', hp.Discrete([0.1])),
    hp.HParam('n_neurons_1', hp.Discrete([25, 50, 75, 100, 125, 150, 175, 200])),
    hp.HParam('dropout_1', hp.Discrete([0.3])),
    hp.HParam('dropout_2', hp.Discrete([0.0])),
    hp.HParam('decay_train_mode', hp.Discrete([1])),
    hp.HParam('threshold_train_mode', hp.Discrete([1])),
    hp.HParam('seed', hp.Discrete([1]))
]

METRICS = [
    hp.Metric("epoch_sparse_categorical_accuracy", group="train", display_name="accuracy (train)",),
    hp.Metric("epoch_loss", group="train", display_name="loss (train)",),
    hp.Metric("epoch_sparse_categorical_accuracy", group="validation", display_name="accuracy (val.)",),
    hp.Metric("epoch_loss", group="validation", display_name="loss (val.)",),
    hp.Metric("info_n_trainable", display_name="n_trainable",),
    hp.Metric("best_val", display_name="best_val",),
    hp.Metric("best_val_arg", display_name="best_val_arg",),
]

def create_model(shape_in, n_out, hparams):
    tau_v = 10.0

    inputs = tf.keras.layers.Input(shape=shape_in)
    mid = tf.keras.layers.MaxPooling3D(pool_size=(1,1,hparams['pool_size']))(inputs)
    mid = wta_layer.WTA(threshold=hparams['wta_threshold'])(mid)
    mid = tf.keras.layers.Reshape((shape_in[0], -1))(mid)
    mid = tf.keras.layers.Dropout(hparams['dropout_1'])(mid)

    mid_z, v = tf.keras.layers.RNN(lif_neuron.RecurrentLifNeuronCell(
        hparams['n_neurons_1'],
        decay_train_mode=hparams['decay_train_mode'],
        threshold_train_mode=hparams['threshold_train_mode'],
        tau=tau_v,
    ), return_sequences=True, name='LIF_recurrent_01')(mid)

    out_z, v = tf.keras.layers.RNN(lif_neuron.LiNeuronCell(
        n_out,
    ), return_sequences=False, name='out')(mid_z)

    model = tf.keras.Model(inputs=inputs, outputs=[out_z])

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"],
        run_eagerly=False
    )

    return model