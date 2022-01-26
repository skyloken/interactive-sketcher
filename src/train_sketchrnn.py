import glob
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras as K
from tqdm import tqdm, trange

print("tf: {}".format(tf.version.VERSION))
print("tf.keras: {}".format(K.__version__))
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))

sys.path.append('../SketchRNN_tf2')
from sketchrnn import dataset, models, utils

with open('../outputs/labels.txt', 'r') as f:
    classes = list(map(lambda s: s.strip(), f.readlines()))

for data_class in classes:

    print("Training:", data_class)

    data = np.load(
        f'../data/quickdraw/{data_class}.npz', encoding='latin1', allow_pickle=True)

    data_train = [dataset.cleanup(d) for d in data['train']]
    data_valid = [dataset.cleanup(d) for d in data['valid']]
    data_test = [dataset.cleanup(d) for d in data['test']]

    hps = {
        "max_seq_len": max(map(len, np.concatenate([data['train'], data['valid'], data['test']]))),
        'batch_size': 100,
        "num_batches": math.ceil(len(data_train) / 100),
        "epochs": 30,
        "recurrent_dropout_prob": 0.1,  # 0.0 for gpu lstm
        "enc_rnn_size": 256,
        "dec_rnn_size": 512,
        "z_size": 128,
        "num_mixture": 20,
        "learning_rate": 0.001,
        "min_learning_rate": 0.00001,
        "decay_rate": 0.9999,
        "grad_clip": 1.0,
        'kl_tolerance': 0.2,
        'kl_decay_rate': 0.99995,
        "kl_weight": 1.0,
        'kl_weight_start': 1.0,
    }

    sketchrnn = models.SketchRNN(hps)
    sketchrnn.models['full'].summary()

    scale_factor = dataset.calc_scale_factor(data_train)

    train_dataset = dataset.make_train_dataset(
        data_train, hps['max_seq_len'], hps['batch_size'], scale_factor)
    val_dataset = dataset.make_val_dataset(
        data_valid, hps['max_seq_len'], hps['batch_size'], scale_factor)

    checkpoint_dir = f'../data/sketchrnn/{data_class}/checkpoints'
    log_dir = f'../data/sketchrnn/{data_class}/logs'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint = os.path.join(checkpoint_dir, 'sketch_rnn_' + data_class + '_weights.{:02d}_{:.2f}.hdf5')
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.hdf5')))
    latest_checkpoint = checkpoints[-1]
    initial_epoch = len(checkpoints)

    sketchrnn.load_weights(latest_checkpoint)
    sketchrnn.train(initial_epoch, train_dataset, val_dataset, checkpoint)
