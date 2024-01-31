import tensorflow as tf


class ShakespeareModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
          rnn_units,
          return_sequences=True,
          return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
          states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
          return x, states
        else:
          return x


def split_input_target(sequence):
    """Split sequence data into (input, label) for model
    training
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def text_from_ids(ids, chars_from_ids_function):
    """Go from numerical IDs to text"""
    return tf.strings.reduce_join(
        chars_from_ids_function(ids), axis=-1
    )
