import json
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

sys.path.append("../sketchformer")
from basic_usage.sketchformer import continuous_embeddings

# load dataset
with open('../data/isketcher/train.json', 'r') as f:
    train = json.load(f)
with open('../data/isketcher/valid.json', 'r') as f:
    valid = json.load(f)
with open('../data/isketcher/test.json', 'r') as f:
    test = json.load(f)

print(f"train: {len(train)}, valid: {len(valid)}, test: {len(test)}")


# load class label
df = pd.read_csv('../outputs/sketchyscene_quickdraw.csv')
df = df.dropna(subset=['quickdraw_label'])
class_names = ['none']
for row in df.itertuples():
    class_names.append(row.quickdraw_label)
class_to_num = dict(zip(class_names, range(0, len(class_names))))

print(len(class_names))
print(class_names)
print(class_to_num)


# load sketchformer
sketchformer = continuous_embeddings.get_pretrained_model()


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


# define preprocess
def preprocess(dataset, datatype, n=1, seed=100):
    input_batch = []
    label_batch = []
    rng = np.random.default_rng(seed)
    for scene in tqdm(dataset):
        for _ in range(n):
            input_scene = []
            labels = []
            sketches = []
            for o in scene:
                sketch = rng.choice(quickdraw_map[o["label"]][datatype])
                sketches.append(sketch)
            sketch_embeddings = sketchformer.get_embeddings(sketches)
            for se, obj in zip(sketch_embeddings, scene):
                p = list(map(lambda x: x / 750, obj['position']))
                o = se.numpy().tolist() + p
                input_scene.append(o)  # オブジェクトの数が不規則
                labels.append(class_to_num[obj['label']])  # convert to num
            indices = rng.permutation(len(scene))
            input_scene = np.array(input_scene)
            input_scene = input_scene[indices]
            labels = np.array(labels)
            labels = labels[indices]
            input_batch.append(input_scene)
            label_batch.append(labels)
    return tf.ragged.constant(input_batch).to_tensor(0.), tf.ragged.constant(label_batch).to_tensor(0)


# preprocess
N = 10
print("Preprocessing train dataset")
x_train, y_train = preprocess(train, "train", N)
print("Preprocessing valid dataset")
x_valid, y_valid = preprocess(valid, "valid", N)
print("Preprocessing test dataset")
x_test, y_test = preprocess(test, "test", N)

np.savez_compressed(f'../data/isketcher/dataset_{N}.npz', x_train=x_train, y_train=y_train,
                    x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test)


dataset = np.load(f'../data/isketcher/dataset_{N}.npz')
print(dataset.files)
print(dataset['x_train'].shape)
print(dataset['y_train'].shape)
print(dataset['x_valid'].shape)
print(dataset['y_valid'].shape)
print(dataset['x_test'].shape)
print(dataset['y_test'].shape)
