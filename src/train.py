import sys
import time

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from isketcher import InteractiveSketcher
from mask import create_combined_mask

sys.path.append("../sketchformer")


# HParams
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([8, 16, 32, 64]))
HP_NUM_LAYERS = hp.HParam('num_units', hp.Discrete([6, 12, 24, 48]))
HP_DFF = hp.HParam('dff', hp.Discrete([512, 1024, 2048, 4096]))
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([5, 10, 13, 26]))

METRIC_LOSS = 'loss/train'
METRIC_VAL_LOSS = 'loss/valid'
METRIC_ACC = 'acc/train'
METRIC_VAL_ACC = 'acc/valid'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_BATCH_SIZE, HP_NUM_LAYERS, HP_DFF, HP_NUM_HEADS],
        metrics=[hp.Metric(METRIC_LOSS, display_name='loss'),
                 hp.Metric(METRIC_VAL_LOSS, display_name='val_loss'),
                 hp.Metric(METRIC_ACC, display_name='acc'),
                 hp.Metric(METRIC_VAL_ACC, display_name='val_acc')],
    )


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


def loss_function(c_real, x_real, y_real, c_pred, x_pred, y_pred):
    scc = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

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


def train_step(interactive_sketcher, optimizer, tar, labels, train_loss, train_accuracy):
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


def valid_step(interactive_sketcher, tar, labels, valid_loss, valid_accuracy):
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


def run(run_dir, run_name, hparams):

    # hyper parameters
    EPOCHS = 1
    # BATCH_SIZE = 16
    # num_layers = 6
    # dff = 2048
    # num_heads = 10
    dropout_rate = 0.1

    # constant
    d_model = 130
    target_object_num = 40  # object num

    # optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    # create model
    interactive_sketcher = InteractiveSketcher(
        num_layers=hparams[HP_NUM_LAYERS], d_model=d_model, num_heads=hparams[HP_NUM_HEADS], dff=hparams[HP_DFF],
        object_num=target_object_num, pe_target=100, rate=dropout_rate)

    # checkpoint
    checkpoint_path = f"./checkpoints/{run_name}"

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                               transformer=interactive_sketcher,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_class_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='valid_class_accuracy')

    # load dataset
    dataset = np.load('../data/isketcher/dataset.npz')
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_valid, y_valid = dataset['x_valid'], dataset['y_valid']

    # tensorboard
    summary_writer = tf.summary.create_file_writer(run_dir)

    for epoch in range(int(ckpt.epoch), EPOCHS + int(ckpt.epoch)):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

        # train
        for i, (x_batch, y_batch) in enumerate(zip(batch(x_train, hparams[HP_BATCH_SIZE]), batch(y_train, hparams[HP_BATCH_SIZE]))):
            train_step(interactive_sketcher, optimizer, x_batch,
                       y_batch, train_loss, train_accuracy)

            if (i + 1) % 100 == 0:
                print('Epoch {}, Batch {}, Loss {:.4f}, Accuracy {:.4f}'.format(
                    epoch, i + 1, train_loss.result(), train_accuracy.result()))

        # valid
        valid_step(interactive_sketcher, x_valid,
                   y_valid, valid_loss, valid_accuracy)

        with summary_writer.as_default():
            tf.summary.scalar(METRIC_LOSS, train_loss.result(), step=epoch)
            tf.summary.scalar(METRIC_ACC, train_accuracy.result(), step=epoch)
            tf.summary.scalar(METRIC_VAL_LOSS, valid_loss.result(), step=epoch)
            tf.summary.scalar(
                METRIC_VAL_ACC, valid_accuracy.result(), step=epoch)

        ckpt.epoch.assign_add(1)
        if epoch % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                epoch, ckpt_save_path))

        print('Epoch {}, Loss {:.4f}, Accuracy {:.4f}, Valid Loss {:.4f}, Valid Accuracy {:.4f}'.format(
            epoch, train_loss.result(), train_accuracy.result(), valid_loss.result(), valid_accuracy.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    with summary_writer.as_default():
        hp.hparams(hparams)


def main():

    session_num = 0

    for batch_size in HP_BATCH_SIZE.domain.values:
        for num_layers in HP_NUM_LAYERS.domain.values:
            for dff in HP_DFF.domain.values:
                for num_heads in HP_NUM_HEADS.domain.values:
                    hparams = {
                        HP_BATCH_SIZE: batch_size,
                        HP_NUM_LAYERS: num_layers,
                        HP_DFF: dff,
                        HP_NUM_HEADS: num_heads,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('logs/hparam_tuning/' + run_name, run_name, hparams)
                    session_num += 1


if __name__ == "__main__":
    main()
