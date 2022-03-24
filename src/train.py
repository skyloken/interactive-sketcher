import datetime
import sys
import time

import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from isketcher import InteractiveSketcher
from loss import giou_loss
from mask import create_combined_mask
from metrics import mean_iou

sys.path.append("../sketchformer")


# HParams
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([128]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([6]))
HP_D_MODEL = hp.HParam('d_model', hp.Discrete([512]))
HP_DFF = hp.HParam('dff', hp.Discrete([2048]))
HP_NUM_HEADS = hp.HParam('num_heads', hp.Discrete([8]))

METRIC_LOSS = 'loss/train'
METRIC_VAL_LOSS = 'loss/valid'
METRIC_PLOSS = 'ploss/train'
METRIC_VAL_PLOSS = 'ploss/valid'
METRIC_CLOSS = 'closs/train'
METRIC_VAL_CLOSS = 'closs/valid'
METRIC_ACC = 'acc/train'
METRIC_VAL_ACC = 'acc/valid'
METRIC_MAE = 'mae/train'
METRIC_VAL_MAE = 'mae/valid'
METRIC_IOU = 'iou/train'
METRIC_VAL_IOU = 'iou/valid'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"./logs/{current_time}"
is_restore = True

with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_BATCH_SIZE, HP_NUM_LAYERS,
                 HP_D_MODEL, HP_DFF, HP_NUM_HEADS],
        metrics=[
            hp.Metric(METRIC_LOSS, display_name='loss'),
            hp.Metric(METRIC_VAL_LOSS, display_name='val_loss'),
            hp.Metric(METRIC_ACC, display_name='acc'),
            hp.Metric(METRIC_VAL_ACC, display_name='val_acc'),
            hp.Metric(METRIC_MAE, display_name='mae'),
            hp.Metric(METRIC_VAL_MAE, display_name='val_mae'),
            hp.Metric(METRIC_IOU, display_name='iou'),
            hp.Metric(METRIC_VAL_IOU, display_name='val_iou')
        ],
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


def loss_function(c_real, p_real, c_pred, p_pred, mask, lc=1.0, lp=1.0):
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()
    sl1 = tf.keras.losses.Huber(delta=1.0)

    # class loss
    # クラスラベルはカテゴリカルクロスエントロピー
    c_loss = scce(c_real, c_pred, sample_weight=mask)

    # position loss
    # 位置座標は平均二乗誤差
    # p_loss = mse(p_real, p_pred, sample_weight=mask)
    # p_loss = sl1(p_real, p_pred, sample_weight=mask)
    p_loss = sl1(p_real[:, :, 0:2], p_pred[:, :, 0:2], sample_weight=mask) + \
        sl1(p_real[:, :, 2:4], p_pred[:, :, 2:4], sample_weight=mask)
    # p_loss = giou_loss(p_real, p_pred, mask)

    return (lc * c_loss) + (lp * p_loss), c_loss, p_loss


def train_step(interactive_sketcher, optimizer, tar, labels, lc, lp, train_loss, train_closs, train_ploss, train_accuracy, train_mae, train_iou):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    p_real = tar_real[:, :, -4:]

    labels_inp = labels[:, :-1]
    labels_real = labels[:, 1:]

    # パディングしたオブジェクトの位置はlabelsが0の位置のため、そこからマスクを作成
    combined_mask = create_combined_mask(labels_inp)

    mask = tf.math.logical_not(tf.math.equal(labels_real, 0))

    with tf.GradientTape() as tape:
        c_out, p_out, _ = interactive_sketcher(
            tar_inp, True, combined_mask)

        loss, c_loss, p_loss = loss_function(
            labels_real, p_real, c_out, p_out, mask, lc, lp)

    gradients = tape.gradient(loss, interactive_sketcher.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, interactive_sketcher.trainable_variables))

    train_loss(loss)
    train_closs(c_loss)
    train_ploss(p_loss)
    train_accuracy(labels_real, c_out, sample_weight=mask)
    train_mae(p_real, p_out)
    train_iou(mean_iou(p_real, p_out, mask))


def valid_step(interactive_sketcher, tar, labels, valid_loss, valid_closs, valid_ploss, valid_accuracy, valid_mae, valid_iou):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    p_real = tar_real[:, :, -4:]

    labels_inp = labels[:, :-1]
    labels_real = labels[:, 1:]

    # パディングしたオブジェクトの位置はlabelsが0の位置のため、そこからマスクを作成
    combined_mask = create_combined_mask(labels_inp)

    c_out, p_out, _ = interactive_sketcher(
        tar_inp, False, combined_mask)

    mask = tf.math.logical_not(tf.math.equal(labels_real, 0))
    loss, c_loss, p_loss = loss_function(
        labels_real, p_real, c_out, p_out, mask)

    valid_loss(loss)
    valid_closs(c_loss)
    valid_ploss(p_loss)
    valid_accuracy(labels_real, c_out, sample_weight=mask)
    valid_mae(p_real, p_out)
    valid_iou(mean_iou(p_real, p_out, mask))


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def run(run_dir, hparams, dataset):

    # hyper parameters
    EPOCHS = 3000
    BATCH_SIZE = hparams[HP_BATCH_SIZE]

    num_layers = hparams[HP_NUM_LAYERS]
    d_model = hparams[HP_D_MODEL]
    dff = hparams[HP_DFF]
    num_heads = hparams[HP_NUM_HEADS]
    dropout_rate = 0.1
    is_shuffle = True

    lc = 0.01
    lp = 1.0

    # constant
    target_object_num = 41  # object num, オブジェクト数は40だがID=0があるため+1

    # optimizer
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    # create model
    interactive_sketcher = InteractiveSketcher(
        num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
        object_num=target_object_num, rate=dropout_rate)

    # checkpoint
    checkpoint_path = "./checkpoints"

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(1),
                               transformer=interactive_sketcher,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=None)

    if is_restore and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_closs = tf.keras.metrics.Mean(name='train_closs')
    train_ploss = tf.keras.metrics.Mean(name='train_ploss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_acc')
    train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    train_iou = tf.keras.metrics.Mean(name='train_loss')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_closs = tf.keras.metrics.Mean(name='valid_closs')
    valid_ploss = tf.keras.metrics.Mean(name='valid_ploss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='valid_acc')
    valid_mae = tf.keras.metrics.MeanAbsoluteError(name='valid_mae')
    valid_iou = tf.keras.metrics.Mean(name='valid_loss')

    # load dataset
    x_train, y_train = dataset['x_train'], dataset['y_train']
    x_valid, y_valid = dataset['x_valid'], dataset['y_valid']

    # tensorboard
    summary_writer = tf.summary.create_file_writer(run_dir)

    # tf.function
    train_step_ = tf.function(train_step)
    valid_step_ = tf.function(valid_step)

    rng = np.random.default_rng(100)

    for epoch in range(int(ckpt.epoch), EPOCHS + int(ckpt.epoch)):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        train_mae.reset_states()
        train_iou.reset_states()

        valid_loss.reset_states()
        valid_accuracy.reset_states()
        valid_mae.reset_states()
        valid_iou.reset_states()

        # shuffle
        if is_shuffle:
            indices = rng.permutation(len(x_train))
            x_train = x_train[indices]
            y_train = y_train[indices]

        # train
        for i, (x_batch, y_batch) in enumerate(zip(batch(x_train, BATCH_SIZE), batch(y_train, BATCH_SIZE))):
            train_step_(interactive_sketcher, optimizer, x_batch,
                        y_batch, lc, lp, train_loss, train_closs, train_ploss, train_accuracy, train_mae, train_iou)

            if (i + 1) % 50 == 0:
                print('Epoch {}, Batch {}, loss {:.4f}, acc {:.4f}, mae {:.4f}, iou {:.4f}'.format(
                    epoch, i + 1, train_loss.result(), train_accuracy.result(), train_mae.result(), train_iou.result()))

        # valid
        valid_step_(interactive_sketcher, x_valid,
                    y_valid, valid_loss, valid_closs, valid_ploss, valid_accuracy, valid_mae, valid_iou)

        with summary_writer.as_default():
            tf.summary.scalar(METRIC_LOSS, train_loss.result(), step=epoch)
            tf.summary.scalar(METRIC_CLOSS, train_closs.result(), step=epoch)
            tf.summary.scalar(METRIC_PLOSS, train_ploss.result(), step=epoch)
            tf.summary.scalar(METRIC_ACC, train_accuracy.result(), step=epoch)
            tf.summary.scalar(METRIC_MAE, train_mae.result(), step=epoch)
            tf.summary.scalar(METRIC_IOU, train_iou.result(), step=epoch)

            tf.summary.scalar(METRIC_VAL_LOSS, valid_loss.result(), step=epoch)
            tf.summary.scalar(METRIC_VAL_CLOSS,
                              valid_closs.result(), step=epoch)
            tf.summary.scalar(METRIC_VAL_PLOSS,
                              valid_ploss.result(), step=epoch)
            tf.summary.scalar(
                METRIC_VAL_ACC, valid_accuracy.result(), step=epoch)
            tf.summary.scalar(METRIC_VAL_MAE, valid_mae.result(), step=epoch)
            tf.summary.scalar(METRIC_VAL_IOU, valid_iou.result(), step=epoch)

        ckpt.epoch.assign_add(1)
        if epoch % 10 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                epoch, ckpt_save_path))

        print('Epoch {}, loss {:.4f}, acc {:.4f}, mae {:.4f}, iou {:.4f}, val_loss {:.4f}, val_acc {:.4f}, val_mae {:.4f}, val_iou {:.4f}'.format(
            epoch, train_loss.result(), train_accuracy.result(), train_mae.result(), train_iou.result(), valid_loss.result(), valid_accuracy.result(), valid_mae.result(), valid_iou.result()))

        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    with summary_writer.as_default():
        hp.hparams(hparams)


def main():

    dataset = np.load('../data/isketcher/dataset_ar.npz')

    session_num = 0
    for batch_size in HP_BATCH_SIZE.domain.values:
        for num_layers in HP_NUM_LAYERS.domain.values:
            for d_model in HP_D_MODEL.domain.values:
                for dff in HP_DFF.domain.values:
                    for num_heads in HP_NUM_HEADS.domain.values:
                        hparams = {
                            HP_BATCH_SIZE: batch_size,
                            HP_NUM_LAYERS: num_layers,
                            HP_D_MODEL: d_model,
                            HP_DFF: dff,
                            HP_NUM_HEADS: num_heads
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run(f'{log_dir}/{run_name}', hparams, dataset)
                        session_num += 1


if __name__ == "__main__":
    main()
