import tensorflow as tf
from pathlib import Path
import os
import argparse
from shakespeare_model import (
    split_input_target, ShakespeareModel
)

# SEQ_LENGTH = 100
SEQ_LENGTH = 10
BATCH_SIZE = 64
BUFFER_SIZE = 10000


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', type=str,
        default=os.environ['SM_CHANNEL_TRAINING']
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=os.environ['SM_MODEL_DIR']
    )
    args = parser.parse_args()

    train_data_path = str(Path(args.train) / 'shakespeare.txt')
    
    text = open(train_data_path, 'rb').read().decode(
        encoding='utf-8'
    )
    vocab = sorted(set(text))

    # Get ids from chars and reversed
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True, mask_token=None
    )

    # Data preparation into sequences
    shakespeare_ids = ids_from_chars(
        tf.strings.unicode_split(text, 'UTF-8')
    )

    ids_dataset = tf.data.Dataset.from_tensor_slices(
        shakespeare_ids
    )

    sequences = ids_dataset.batch(
        SEQ_LENGTH+1, drop_remainder=True
    )

    dataset = sequences.map(split_input_target)

    # Create training batches
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

    loss = tf.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )
    
    EPOCHS = 1
    history = model.fit(
        dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback]
    )

    # Save model to S3
    # model_path = str(
    #     Path(args.model_dir) / 'model.keras'
    # )
    model.save(args.model_dir)
