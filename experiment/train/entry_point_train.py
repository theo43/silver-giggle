import tensorflow as tf
import numpy as np
import os



SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000


def text_from_ids(ids):
    """Go from numerical IDs to text"""
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def split_input_target(sequence):
    """Split sequence data into (input, label) for model training
    """
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


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


if __name__ == '__main__':
    
    train_data_path = os.environ['SM_CHANNEL_TRAINING']

    text = open(train_data_path, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))

    # get ids from chars and reversed
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True, mask_token=None
    )

    # data preparation into sequences
    shakespeare_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

    ids_dataset = tf.data.Dataset.from_tensor_slices(shakespeare_ids)

    sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    # create training batches
    dataset = dataset.shuffle(BUFFER_SIZE)\
        .batch(BATCH_SIZE, drop_remainder=True)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = ShakespeareModel(
       vocab_size=vocab_size,
       embedding_dim=embedding_dim,
       rnn_units=rnn_units
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    EPOCHS = 1
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    
