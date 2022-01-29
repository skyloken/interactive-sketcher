import glob
import math
import random
import sys

import numpy as np
import tensorflow as tf

from isketcher import InteractiveSketcher
from util import to_normal_strokes

if 1 == 1:
    sys.path.append("../sketchformer")
    from basic_usage.sketchformer import continuous_embeddings
    sys.path.append('../SketchRNN_tf2')
    from sketchrnn import dataset, models

CANVAS_SIZE = 750

# class names
with open('../outputs/labels.txt', 'r') as f:
    class_names = list(map(lambda s: s.strip(), f.readlines()))


class Agent:

    def __init__(self) -> None:

        # Hyper parameters
        num_layers = 6
        d_model = 512
        dff = 2048
        num_heads = 8
        dropout_rate = 0.1
        target_object_num = 41

        self.interactive_sketcher = InteractiveSketcher(
            num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
            object_num=target_object_num, rate=dropout_rate)

        # restore model
        ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                                   transformer=self.interactive_sketcher)

        checkpoint_path = "../models/isketcher/model_12/checkpoints_3000/"
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
            print(ckpt_manager.latest_checkpoint)

    def get_next_sketch(self, inp):
        c_out, p_out, _ = self.interactive_sketcher(
            inp, training=False, look_ahead_mask=None)

        c_pred = c_out[0, -1, :]  # 最後のスケッチを取得
        c_pred_id = tf.argmax(c_pred, axis=-1)
        position = p_out[0, -1, :] * CANVAS_SIZE  # 最後のスケッチを取得

        return {
            "name": class_names[c_pred_id - 1],
            "position": list(map(int, position.numpy().tolist()))
        }

    def get_rand_sketch(self, min_w=30, min_h=30, max_w=CANVAS_SIZE, max_h=CANVAS_SIZE):
        c = random.choice(class_names)
        w = random.randint(min_w, max_w)
        h = random.randint(min_h, max_h)
        min_x = w // 2
        min_y = h // 2
        max_x = CANVAS_SIZE - min_x
        max_y = CANVAS_SIZE - min_y
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        return {
            "name": c,
            "position": [x, y, w, h]
        }


class Sketchformer:

    def __init__(self) -> None:
        self.sketchformer = continuous_embeddings.get_pretrained_model()

    def preprocess(self, sketches):
        inp = []

        strokes_list = list(map(lambda o: o["strokes"], sketches))
        sketch_embeddings = self.sketchformer.get_embeddings(strokes_list)
        for se, obj in zip(sketch_embeddings, sketches):
            p = list(map(lambda x: x / CANVAS_SIZE, obj["position"]))
            o = se.numpy().tolist() + p
            inp.append(o)

        return tf.ragged.constant([inp]).to_tensor(0.)

    def classify(self, sketches):
        return self.sketchformer.classify(sketches)


class SketchRNN:

    def __init__(self) -> None:
        self.from_dataset = False
        self.quickdraw = {}

        for name in class_names:
            quickdraw_npz = np.load(
                f'../data/quickdraw/{name}.npz', encoding='latin1', allow_pickle=True)
            self.quickdraw[name] = {
                "train": quickdraw_npz["train"],
                "valid": quickdraw_npz["valid"],
                "test": quickdraw_npz["test"]
            }

    def get_random_strokes(self, name, temp=0.01):
        if self.from_dataset:
            return random.choice(self.quickdraw[name]["test"]).tolist()
        else:
            data = self.quickdraw[name]
            data_train = [dataset.cleanup(d) for d in data['train']]

            hps = {
                "max_seq_len": max(map(len, np.concatenate([data['train'], data['valid'], data['test']]))),
                'batch_size': 100,
                "num_batches": math.ceil(len(data_train) / 100),
                "epochs": 100,
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
                "kl_weight": 0.5,
                'kl_weight_start': 0.01,
            }

            sketchrnn = models.SketchRNN(hps)

            checkpoint = sorted(
                glob.glob(f'../data/sketchrnn/{name}/checkpoints/*.hdf5'))[-1]
            sketchrnn.load_weights(checkpoint)
            strokes = sketchrnn.sample(temperature=temp)
            normal_strokes = to_normal_strokes(strokes)

            return normal_strokes.tolist()
