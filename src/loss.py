import tensorflow as tf
import tensorflow_addons as tfa


def giou_loss(y_true, y_pred, mask):
    gl = tfa.losses.GIoULoss()

    y_true = tf.concat([y_true[:, :, :2] - y_true[:, :, 2:] * 0.5,
                        y_true[:, :, :2] + y_true[:, :, 2:] * 0.5], axis=-1)
    y_true = tf.concat([y_true[:, :, 1:2], y_true[:, :, 0:1],
                       y_true[:, :, 3:4], y_true[:, :, 2:3]], axis=-1)

    y_pred = tf.concat([y_pred[:, :, :2] - y_pred[:, :, 2:] * 0.5,
                        y_pred[:, :, :2] + y_pred[:, :, 2:] * 0.5], axis=-1)
    y_pred = tf.concat([y_pred[:, :, 1:2], y_pred[:, :, 0:1],
                       y_pred[:, :, 3:4], y_pred[:, :, 2:3]], axis=-1)

    loss = gl(y_true, y_pred, sample_weight=mask)
    return loss
