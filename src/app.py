import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request

from isketcher import InteractiveSketcher
from util import (adjust_lines, draw_strokes, lines_to_sketch,
                  strokes_to_lines, visualize)

sys.path.append("../sketchformer")
from basic_usage.sketchformer import continuous_embeddings

app = Flask(__name__)
CANVAS_SIZE = 750


def get_model(checkpoint_path):

    num_layers = 6
    d_model = 512
    dff = 2048
    num_heads = 8
    dropout_rate = 0.1
    target_object_num = 41

    interactive_sketcher = InteractiveSketcher(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        object_num=target_object_num, rate=dropout_rate)

    # restore model

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                               transformer=interactive_sketcher)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    return interactive_sketcher


# setup
sketchformer = continuous_embeddings.get_pretrained_model()
interactive_sketcher = get_model("../models/model_12/checkpoints/")

# class label
df = pd.read_csv('../outputs/sketchyscene_quickdraw.csv')
df = df.dropna(subset=['quickdraw_label'])
class_names = ['none']
for row in df.itertuples():
    class_names.append(row.quickdraw_label)

quickdraw_map = {}
df = pd.read_csv('../outputs/sketchyscene_quickdraw.csv')
df = df.dropna(subset=['quickdraw_label'])
for row in df.itertuples():
    quickdraw = np.load(
        f'../data/sketch_rnn/{row.quickdraw_label}.npz', encoding='latin1', allow_pickle=True)
    quickdraw_map[row.quickdraw_label] = {
        "train": quickdraw["train"],
        "valid": quickdraw["valid"],
        "test": quickdraw["test"],
    }


def preprocess(sketches):
    inp = []

    strokes_list = list(map(lambda o: o["strokes"], sketches))
    sketch_embeddings = sketchformer.get_embeddings(strokes_list)
    for se, obj in zip(sketch_embeddings, sketches):
        p = list(map(lambda x: x / CANVAS_SIZE, obj["position"]))
        o = se.numpy().tolist() + p
        inp.append(o)

    return tf.ragged.constant([inp]).to_tensor(0.)


def get_random_strokes(name):
    return random.choice(quickdraw_map[name]["test"]).tolist()


def get_next_sketch(inp):
    c_out, p_out, _ = interactive_sketcher(
        inp, training=False, look_ahead_mask=None)

    c_pred = c_out[0, -1, :]  # 最後のスケッチを取得
    c_pred_id = tf.argmax(c_pred, axis=-1)
    position = p_out[0, -1, :] * CANVAS_SIZE  # 最後のスケッチを取得

    return {
        "name": class_names[c_pred_id],
        "position": list(map(int, position.numpy().tolist()))
    }


@app.route("/api/sketch", methods=["POST"])
def draw_next_sketch():
    previous_sketches = request.get_json()["previousSketches"]
    user_lines = request.get_json()["userLines"]

    # convert to stroke-3 format
    user_sketch = lines_to_sketch(user_lines)

    # visualize
    # visualize(user_sketch["strokes"])
    # draw_strokes(user_sketch["strokes"])

    # next sketch
    inp = preprocess(previous_sketches + [user_sketch])
    next_sketch = get_next_sketch(inp)

    # TODO: Sketch-RNNから生成する
    strokes = get_random_strokes(next_sketch["name"])
    agent_sketch = {
        "strokes": strokes,
        "position": next_sketch["position"]
    }

    lines = strokes_to_lines(strokes)
    adjusted_lines = adjust_lines(lines, next_sketch["position"])

    # predict
    pred_class = sketchformer.classify([user_sketch["strokes"]])
    print(pred_class)

    return jsonify({
        "nextSketch": next_sketch,
        "nextLines": adjusted_lines,
        "previousSketches": previous_sketches + [user_sketch] + [agent_sketch]
    })


if __name__ == "__main__":
    app.run(debug=True)
