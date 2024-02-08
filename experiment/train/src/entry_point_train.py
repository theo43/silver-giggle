import tensorflow as tf
import numpy as np
from pathlib import Path
import os
import argparse
from shakespeare_model import (
    ShakespeareModel, OneStepModel, generate_batch_dataset
)
import pickle
import boto3
#from aws_utils import upload_folder_to_s3


SEQ_LENGTH = 100
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

    # Load train text data
    train_data_path = str(Path(args.train) / 'train_text.txt')
    text = open(train_data_path, 'rb').read().decode(
        encoding='utf-8'
    )
    # vocab = sorted(set(text))
    
    # Load vocabulary of the whole text
    vocab_path = str(Path(args.train) / 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Reduce data amount for smoke training
    text = text[:10000]

    # Get ids from chars and reversed
    ids_from_chars = tf.keras.layers.StringLookup(
        vocabulary=list(vocab), mask_token=None
    )
    chars_from_ids = tf.keras.layers.StringLookup(
        vocabulary=ids_from_chars.get_vocabulary(),
        invert=True, mask_token=None
    )

    train_dataset_batch = generate_batch_dataset(
        text=text,
        ids_from_chars=ids_from_chars,
        seq_length=SEQ_LENGTH,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE
    )

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
        train_dataset_batch,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback]
    )

    # Integrate it to final custom model
    one_step_model = OneStepModel(
        model=model,
        chars_from_ids=chars_from_ids,
        ids_from_chars=ids_from_chars
    )
    # Call build to initialize computational graph
    for input_example, _ in train_dataset_batch.take(1):
        one_step_model.build(input_example.shape)
    print(one_step_model.summary())

    # Save it locally
    model_local_path = f'{os.environ["SM_MODEL_DIR"]}/one_step_model.keras'
    one_step_model.save(model_local_path)

    # Reload it to check it behaves as the saved one
    one_step_model_loaded = tf.keras.models.load_model(model_local_path)
    # Check that shape and all elements are equal
    np.testing.assert_allclose(
        one_step_model.model.predict(input_example),
        one_step_model_loaded.model.predict(input_example)
    )

    # Then send it to s3
    bucket_name = args.model_dir.split('://')[1].split('/')[0]
    destination_list = args.model_dir.split('://')[1].split('/')[:1]
    destination = '/'.join(destination_list)
    print(f'MODEL_DIR: {args.model_dir}')
    print(f'DESTINATION: {destination}')
    s3_client = boto3.client('s3')
    s3_client.upload_file(model_local_path, bucket_name, destination)
