import tensorflow as tf

from layers import DecoderOnly


class InteractiveSketcher(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        object_num,
        pe_target,
        rate=0.1,
    ):
        super(InteractiveSketcher, self).__init__()

        self.object_num = object_num

        self.decoder = DecoderOnly(
            num_layers, d_model, num_heads, dff, object_num, pe_target, rate, False, False
        )

        self.final_layer = tf.keras.layers.Dense(
            object_num + 2)  # OBJECT_NUM (40) + POSITION (2)

    def call(
        self, tar, training, look_ahead_mask
    ):

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, training, look_ahead_mask,
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, object_num + x + y)

        c_output = tf.nn.softmax(final_output[:, :, :self.object_num])
        x_output = tf.math.sigmoid(final_output[:, :, -2])
        y_output = tf.math.sigmoid(final_output[:, :, -1])

        return c_output, x_output, y_output, attention_weights
