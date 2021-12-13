import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # アテンション・ロジットにパディングを追加するため
    # さらに次元を追加する
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoderパディング・マスク
    enc_padding_mask = create_padding_mask(inp)

    # デコーダーの 2つ目のアテンション・ブロックで使用
    # このパディング・マスクはエンコーダーの出力をマスクするのに使用
    dec_padding_mask = create_padding_mask(inp)

    # デコーダーの 1つ目のアテンション・ブロックで使用
    # デコーダーが受け取った入力のパディングと将来のトークンをマスクするのに使用
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


def create_combined_mask(tar):

    # デコーダーの 1つ目のアテンション・ブロックで使用
    # デコーダーが受け取った入力のパディングと将来のトークンをマスクするのに使用
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask
