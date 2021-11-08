import tensorflow as tf
import tensorflow_datasets as tfds
from tensorboard.plugins.hparams import api as hp
import numpy as np
import itertools

import random
import os
import importlib
import pathlib
import time

from src.misc import memory_growth, lr_scheduler
from src.datasets import dataset_helper

this_path = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))
default_board_path = this_path / "output"

def get_time_str():
    return time.strftime("%y%m%d_%H%M%S_")

def create_tfboard_name(name, current_time_str=get_time_str()):
    path_str = str(default_board_path/name)

    os.system('rm -rf ' + path_str)
    return path_str

def final_log(logdir, best_test_accuracies, last_test_accuracies):
    best_mean = np.round(np.mean(best_test_accuracies), 4)
    last_mean = np.round(np.mean(last_test_accuracies), 4)
    with open(logdir + '/results.txt', 'a') as f:
        write_str = (
            '\nAccuracies: ' + str(best_test_accuracies)
             + '\nLast: ' + str(last_test_accuracies)
             + '\nBest mean: ' + str(best_mean)
             + '\nLast mean: ' + str(last_mean)
        )
        print(write_str)
        f.write(write_str)


def train(model, ds_train, ds_val, hparams, session_num, logdir, epochs, steps_per_epoch):
    callbacks = []
    callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)) # stop after 6 epochs with # no loss improvement

    total_steps = int(epochs * steps_per_epoch)
    warmup_steps = int(1 * steps_per_epoch)
    fixed_lr_steps = int(0.5 * (epochs-1) * steps_per_epoch)
    warm_up_lr_callback = lr_scheduler.WarmUpCosineDecayScheduler(learning_rate_base=1e-3,
                                    total_steps=total_steps,
                                    warmup_learning_rate=0.0,
                                    warmup_steps=warmup_steps,
                                    hold_base_rate_steps=fixed_lr_steps)
    callbacks.append(warm_up_lr_callback)

    run_name = "/run-%d" % session_num
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=logdir + run_name, histogram_freq=1, profile_batch = 0))
    callbacks.append(hp.KerasCallback(logdir + run_name, hparams))

    history = model.fit(
        ds_train, 
        validation_data=ds_val, 
        epochs=epochs, 
        callbacks=callbacks
    )

    best_val = np.round(np.max(history.history['val_sparse_categorical_accuracy']), 4)
    best_val_arg = np.argmax(history.history['val_sparse_categorical_accuracy'])

    info_n_trainable = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    with tf.summary.create_file_writer(logdir + run_name).as_default():
        tf.summary.scalar("info_n_trainable", info_n_trainable, step=1)
        tf.summary.scalar("best_val", best_val, step=1)
        tf.summary.scalar("best_val_arg", best_val_arg, step=1)

    with open(logdir + '/results.txt', 'a') as f:
        write_str = (
            str(session_num)
            + ': params: ' + str(hparams)
            + ', train: ' + str(np.round(np.max(history.history['sparse_categorical_accuracy']), 4))
            + ', val: ' + str(np.round(history.history['val_sparse_categorical_accuracy'][-1], 4))
            + ', best_val: ' + str(best_val)
            + ', best_val_arg: ' + str(best_val_arg)
            + '\n'
        )
        print(write_str)
        f.write(write_str)

    return history.history['val_sparse_categorical_accuracy']


def cv_loo_tinyradar(FLAGS):
    experiment = importlib.import_module('.' + FLAGS['experiment_name'], 'src.experiments')
    values = (param.domain.values for param in experiment.HPARAMS)
    keys = [param.name for param in experiment.HPARAMS]
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    hparams = combinations[0]

    logdir = create_tfboard_name(FLAGS['experiment_name'])
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=experiment.HPARAMS, metrics=experiment.METRICS)

    n_persons = 26
    max_length = 25
    shape_in = (max_length, 16, 123, 2)
    n_out = 12
    mapper = dataset_helper.Mapping(max_length)

    best_test_accuracies = []
    last_test_accuracies = []

    for session_num in range(n_persons):
        tf.keras.backend.clear_session()
        tf.random.set_seed(hparams['seed'])
        random.seed(hparams['seed'])
        os.environ['PYTHONHASHSEED']=str(hparams['seed'])
        np.random.seed(hparams['seed'])

        train_split, eval_split = dataset_helper.loo_split_aug(session_num, 26, [0,4,8,12])
        print('train', train_split, 'eval', eval_split)

        (ds_train, ds_test), ds_info = tfds.load(
            'tinyradar_dataset',
            shuffle_files=True,
            as_supervised=True, 
            split=[train_split, eval_split], 
            with_info=True)
        ds_train = ds_train.map(mapper.pad_zeros, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(mapper.pad_zeros, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(ds_info.splits[train_split].num_examples / FLAGS['batch_size'])
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits[train_split].num_examples//4)
        ds_train = ds_train.batch(FLAGS['batch_size'])
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.cache()
        ds_test = ds_test.batch(512)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        print('-------------------------------------------------')
        print('Session num ', str(session_num))
        print('params: ', hparams)
        print('-------------------------------------------------')

        model = experiment.create_model(shape_in, n_out, hparams)
        model.summary()

        test_acc = train(model, ds_train, ds_test, hparams, session_num, logdir, FLAGS['epochs'], steps_per_epoch)

        best_test_accuracies.append(np.round(np.max(test_acc), 4))
        last_test_accuracies.append(np.round(test_acc[-1], 4))

        del ds_train
        del ds_test

    final_log(logdir, best_test_accuracies, last_test_accuracies)

def cv_loo_soli(FLAGS):
    experiment = importlib.import_module('.' + FLAGS['experiment_name'], 'src.experiments')
    values = (param.domain.values for param in experiment.HPARAMS)
    keys = [param.name for param in experiment.HPARAMS]
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    hparams = combinations[0]

    logdir = create_tfboard_name(FLAGS['experiment_name'])
    with tf.summary.create_file_writer(logdir).as_default():
        hp.hparams_config(hparams=experiment.HPARAMS, metrics=experiment.METRICS)

    n_persons = 10
    max_length = 40
    shape_in = (max_length, 32, 32, 4)
    n_out = 11
    mapper = dataset_helper.Mapping(max_length)

    best_test_accuracies = []
    last_test_accuracies = []

    for session_num in range(n_persons):
        tf.keras.backend.clear_session()
        tf.random.set_seed(hparams['seed'])
        random.seed(hparams['seed'])
        os.environ['PYTHONHASHSEED']=str(hparams['seed'])
        np.random.seed(hparams['seed'])

        train_split, eval_split = dataset_helper.loo_split(session_num, n_persons)
        print('train', train_split, 'eval', eval_split)

        (ds_train, ds_test), ds_info = tfds.load(
            'interfacing_soli_dataset',
            shuffle_files=True,
            as_supervised=True, 
            split=[train_split, eval_split], 
            with_info=True)
        ds_train = ds_train.map(mapper.pad_zeros, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(mapper.pad_zeros, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        steps_per_epoch = tf.math.ceil(ds_info.splits[train_split].num_examples / FLAGS['batch_size'])
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits[train_split].num_examples//4)
        ds_train = ds_train.batch(FLAGS['batch_size'])
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        ds_test = ds_test.cache()
        ds_test = ds_test.batch(512)
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        print('-------------------------------------------------')
        print('Session num ', str(session_num))
        print('params: ', hparams)
        print('-------------------------------------------------')

        model = experiment.create_model(shape_in, n_out, hparams)
        model.summary()

        test_acc = train(model, ds_train, ds_test, hparams, session_num, logdir, FLAGS['epochs'], steps_per_epoch)

        best_test_accuracies.append(np.round(np.max(test_acc), 4))
        last_test_accuracies.append(np.round(test_acc[-1], 4))

        del ds_train
        del ds_test

    final_log(logdir, best_test_accuracies, last_test_accuracies)

if __name__ == '__main__':
    FLAGS = {
        'batch_size': 256,
        'epochs': 30,
        'experiment_name': 'final_spiking_loo_wta_010',
        'dataset': 'tinyradar' # or 'soli'
    }

    if FLAGS['dataset'] == 'tinyradar':
        cv_loo_tinyradar(FLAGS)
    else:
        cv_loo_soli(FLAGS)