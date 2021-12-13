import sys
import time

import numpy as np
import tensorflow as tf

from isketcher import InteractiveSketcher
from mask import create_combined_mask

sys.path.append("../sketchformer")


# hyper parameters
num_layers = 4
d_model = 130
dff = 512
num_heads = 5

target_object_num = 40  # object num
dropout_rate = 0.1

EPOCHS = 100
BATCH_SIZE = 16


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# create model
interactive_sketcher = InteractiveSketcher(
    num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
    object_num=target_object_num, pe_target=100, rate=dropout_rate)


# checkpoint
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                           transformer=interactive_sketcher,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# チェックポイントが存在したなら、最後のチェックポイントを復元
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


scc = tf.keras.losses.SparseCategoricalCrossentropy(
    reduction=tf.keras.losses.Reduction.NONE)
mse = tf.keras.losses.MeanSquaredError(
    reduction=tf.keras.losses.Reduction.NONE)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_class_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='valid_class_accuracy')


def loss_function(c_real, x_real, y_real, c_pred, x_pred, y_pred):
    # class loss
    # クラスラベルはカテゴリカルクロスエントロピー
    c_loss_ = scc(c_real, c_pred)

    # position loss
    # 位置座標は平均二乗誤差
    p_loss_ = tf.math.square(x_real - x_pred) + \
        tf.math.square(y_real - y_pred)

    # mask padded object
    # パディングしたオブジェクトの部分を損失に加えないようにマスクする
    mask = tf.math.logical_not(tf.math.equal(c_real, 0))
    c_mask = tf.cast(mask, dtype=c_loss_.dtype)
    c_loss = tf.reduce_mean(c_loss_ * c_mask)
    p_mask = tf.cast(mask, dtype=p_loss_.dtype)
    p_loss = tf.reduce_mean(p_loss_ * p_mask)

    return c_loss + p_loss


def train_step(tar, labels):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    x_real, y_real = tar_real[:, :, -2], tar_real[:, :, -1]

    labels_inp = labels[:, :-1]
    labels_real = labels[:, 1:]

    # パディングしたオブジェクトの位置はlabelsが0の位置のため、そこからマスクを作成
    combined_mask = create_combined_mask(labels_inp)

    with tf.GradientTape() as tape:
        c_out, x_out, y_out, _ = interactive_sketcher(
            tar_inp, True, combined_mask)

        loss = loss_function(labels_real, x_real, y_real, c_out, x_out, y_out)

    gradients = tape.gradient(loss, interactive_sketcher.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, interactive_sketcher.trainable_variables))

    train_loss(loss)
    train_accuracy(labels_real, c_out)


def valid_step(tar, labels):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    x_real, y_real = tar_real[:, :, -2], tar_real[:, :, -1]

    labels_inp = labels[:, :-1]
    labels_real = labels[:, 1:]

    # パディングしたオブジェクトの位置はlabelsが0の位置のため、そこからマスクを作成
    combined_mask = create_combined_mask(labels_inp)

    c_out, x_out, y_out, _ = interactive_sketcher(
        tar_inp, False, combined_mask)

    loss = loss_function(labels_real, x_real, y_real, c_out, x_out, y_out)

    valid_loss(loss)
    valid_accuracy(labels_real, c_out)


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def main():
    # load dataset
    dataset = np.load('../data/isketcher/dataset.npz')
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_valid, y_valid = dataset['x_valid'], dataset['y_valid']

    # tensorboard
    train_summary_writer = tf.summary.create_file_writer('logs/train')
    valid_summary_writer = tf.summary.create_file_writer('logs/valid')

    for epoch in range(int(ckpt.epoch), EPOCHS + int(ckpt.epoch)):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # train
        for i, (x_batch, y_batch) in enumerate(zip(batch(x_train, BATCH_SIZE), batch(y_train, BATCH_SIZE))):
            train_step(x_batch, y_batch)

            if (i + 1) % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch, i + 1, train_loss.result(), train_accuracy.result()))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        # valid
        valid_step(x_valid, y_valid)

        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

        ckpt.epoch.assign_add(1)
        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f} Valid_Loss {:.4f} Valid_Accuracy {:.4f}'.format(
            epoch, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


if __name__ == "__main__":
    main()
