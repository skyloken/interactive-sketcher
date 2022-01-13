from json.tool import main

import numpy as np
import tensorflow as tf


def bboxes_iou(boxes1, boxes2):

    boxes2 = tf.cast(boxes2, boxes1.dtype)

    boxes1 = tf.concat([boxes1[:, :, :2] - boxes1[:, :, 2:] * 0.5,
                        boxes1[:, :, :2] + boxes1[:, :, 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[:, :, :2] - boxes2[:, :, 2:] * 0.5,
                        boxes2[:, :, :2] + boxes2[:, :, 2:] * 0.5], axis=-1)

    boxes1_area = (boxes1[:, :, 2] - boxes1[:, :, 0]) * \
        (boxes1[:, :, 3] - boxes1[:, :, 1])
    boxes2_area = (boxes2[:, :, 2] - boxes2[:, :, 0]) * \
        (boxes2[:, :, 3] - boxes2[:, :, 1])

    left_up = tf.math.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
    right_down = tf.math.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])

    inter_section = tf.math.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[:, :, 0] * inter_section[:, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / union_area

    return ious


def mean_iou(y_true, y_pred, mask=None):
    ious = bboxes_iou(y_true, y_pred)
    if mask != None:
        mask = tf.cast(mask, dtype=ious.dtype)
        ious *= mask
    return tf.reduce_mean(ious)


def main():
    box1 = tf.constant([[[1, 1, 2, 2], [1, 1, 2, 2]], [
                       [1, 1, 2, 2], [1, 1, 2, 2]]], dtype=tf.float32)
    box2 = tf.constant([[[1, 1, 2, 2], [2, 2, 2, 2]], [
                       [2, 2, 2, 2], [1, 1, 2, 2]]], dtype=tf.float32)
    print(box1.shape)
    print(box2.shape)
    ious = bboxes_iou(box1, box2)
    miou = mean_iou(box1, box2)
    print(ious)
    print(miou)


if __name__ == "__main__":
    main()
