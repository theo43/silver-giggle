import torch
import torch.nn as nn
from pathlib import Path
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from datasets import load_dataset
from tokenizers import Tokenizer
from translation.config import get_config
from translation.model import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', type=str,
        default=os.environ['SM_CHANNEL_TRAINING']
    )
    parser.add_argument(
        '--tokenizers', type=str,
        default=os.environ['SM_CHANNEL_TOKENIZERS']
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=os.environ['SM_MODEL_DIR']
    )
    args = parser.parse_args()

    # Create config
    config = get_config()
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']

    ds_raw = load_dataset(
        'opus_books',
        f'{lang_src}-{lang_tgt}',
        split=f'train[:{config["download_size"]}%]'
    )

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Load train dataloader and tokenizers
    train_dataloader = torch.load(
        str(Path(args.train) / 'train_dataloader.pkl'))
    tokenizer_src = Tokenizer.from_file(
        str(Path(args.tokenizers) / f'tokenizer_{lang_src}.json'))
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(args.tokenizers) / f'tokenizer_{lang_tgt}.json'))
    
    warnings.filterwarnings('ignore')

    # Create model
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    
    initial_epoch = 0
    global_step = 0
    
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(
            train_dataloader,
            desc=f'Processing epoch {epoch:02d}'
        )
        for batch in batch_iterator:
            # (batch, seq_len)
            encoder_input = batch['encoder_input'].to(device)

            # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)

            # (batch, 1, 1, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)

            # (batch, 1, seq_len, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)

            # Run the tensors through transformer
            # (batch, seq_len, d_model)
            encoder_output = model.encode(encoder_input, encoder_mask)
            # (batch, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )
            # (batch, seq_len, tgt_vocab_size)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)  # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size)
            # --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1)
            )
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            # Log the loss
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Back propagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

    # Save model
    model_folder = f'{os.environ["SM_MODEL_DIR"]}/{config["model_folder"]}'
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    model_local_path = f'{model_folder}/{config["model_filename"]}{epoch:02d}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'tokenizer_src': tokenizer_src.save_pretrained(model_folder),
        'tokenizer_tgt': tokenizer_tgt.save_pretrained(model_folder)
    }, model_local_path)

    print(f'Model saved at {model_local_path}')

    print('Training finished')
