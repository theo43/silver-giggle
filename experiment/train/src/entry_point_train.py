import tensorflow as tf
from pathlib import Path
import os
import argparse
from shakespeare_model import (
    ShakespeareModel, generate_batch_dataset
)
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

    train_data_path = str(Path(args.train) / 'shakespeare.txt')
    
    text = open(train_data_path, 'rb').read().decode(
        encoding='utf-8'
    )
    vocab = sorted(set(text))

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

    # Save it locally
    model_local_path = './shakespeare_model.keras'
    model.save(model_local_path)

    # Then to s3
    bucket_name = args.model_dir.split('://')[1].split('/')[0]
    destination_list = args.model_dir.split('://')[1].split('/')[:1]
    destination = '/'.join(destination_list)
    s3_client = boto3.client('s3')
    s3_client.upload_file(model_local_path, bucket_name, destination)
    # Save also to model_dir
    #tf.saved_model.save(model, args.model_dir)
    
    # Push model folder to s3
    
    # upload_folder_to_s3(
    #     local_folder_path=model_local_folder,
    #     s3_bucket_name=bucket_name,
    #     path_on_s3=destination
    # )
