import sys

sys.path.append("../sketchformer")

import tensorflow as tf
from basic_usage.sketchformer import continuous_embeddings

from layers import DecoderOnly


class InteractiveSketcher(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        target_vocab_size,
        pe_target,
        rate=0.1,
    ):
        super(InteractiveSketcher, self).__init__()

        self.sketchformer = continuous_embeddings.get_pretrained_model()

        self.decoder = DecoderOnly(
            num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate, False, True
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(
        self, tar, training, look_ahead_mask
    ):

        # sketch_embeddings == (tar_seq_len, 128)
        sketch_embeddings = self.sketchformer.get_embeddings(tar)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            sketch_embeddings, training, look_ahead_mask,
        )

        final_output = self.final_layer(
            dec_output
        )  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
