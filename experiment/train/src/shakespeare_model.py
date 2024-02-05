import tensorflow as tf
from keras.src.layers.preprocessing.string_lookup import StringLookup


tf.config.run_functions_eagerly(True)


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


class ShakespeareModel(tf.keras.Model):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rnn_units: int
    ):
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


class OneStepModel(tf.keras.Model):
    def __init__(
        self,
        model: ShakespeareModel,
        chars_from_ids: StringLookup,
        ids_from_chars: StringLookup,
        temperature: float = 1.0
    ):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(
            inputs=input_ids, states=states, return_state=True
        )
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

    @tf.function
    def generate_one_sentence(self, input_text):
        states = None
        next_char = tf.constant([input_text])
        result = [next_char]
        stop = False

        while not stop:
            next_char, states = self.generate_one_step(
                next_char, states=states
            )
            result.append(next_char)
            char_utf8 = next_char.numpy()[0].decode('utf-8')
            if char_utf8 in ['.', '?', '!', '\n']:
                stop = True

        return tf.strings.join(result)[0].numpy().decode('utf-8')


def generate_batch_dataset(
    text: str,
    ids_from_chars: StringLookup,
    seq_length: int,
    buffer_size: int,
    batch_size: int
):
    # Data preparation into sequences
    ids = ids_from_chars(
        tf.strings.unicode_split(text, 'UTF-8')
    )
    ids_dataset = tf.data.Dataset.from_tensor_slices(ids)
    sequences = ids_dataset.batch(
        seq_length + 1, drop_remainder=True
    )

    # Create dataset from sequence
    dataset = sequences.map(split_input_target)

    # Create training batches
    dataset_batch = dataset.shuffle(buffer_size) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    return dataset_batch
