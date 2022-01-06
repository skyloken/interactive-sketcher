import tensorflow as tf

from layers import DecoderOnlyForSketcher


class InteractiveSketcher(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        object_num,
        rate=0.1,
    ):
        super(InteractiveSketcher, self).__init__()

        self.object_num = object_num

        self.decoder = DecoderOnlyForSketcher(
            num_layers, d_model, num_heads, dff, rate)

        self.final_layer = tf.keras.layers.Dense(object_num + 4)

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
        p_output = tf.math.sigmoid(final_output[:, :, self.object_num:])

        return c_output, p_output, attention_weights
