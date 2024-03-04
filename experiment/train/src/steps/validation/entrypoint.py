import json
import torch
import torch.nn as nn
from pathlib import Path
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer
from translation.config import get_config
from translation.model import get_model


if __name__ == '__main__':

    processing_dir = '/opt/ml/processing'

    # Create config
    config = get_config()
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']

    ds_raw = load_dataset(
        'opus_books',
        f'{lang_src}-{lang_tgt}',
        split=f'train[:{config["download_size"]}%]'
    )

    # Load valid dataloader, tokenizers and model weights
    model_checkpoint = torch.load(
        str(Path(processing_dir) / '/model/weights/tmodel_02.pt'))
    valid_dataloader = torch.load(
        str(Path(processing_dir) / '/valid/valid_dataloader.pkl'))
    tokenizer_src = Tokenizer.from_file(
        str(Path(processing_dir) / f'/tokenizers/tokenizer_{lang_src}.json'))
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(processing_dir) / f'/tokenizers/tokenizer_{lang_tgt}.json'))

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Create model, load previously trained weights and set to eval mode
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    