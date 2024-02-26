import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from translation.dataset import BilingualDataset
from translation.model import build_transformer
from translation.tokenizer import get_or_build_tokenizer
from translation_config import get_weights_file_path, get_config
from pathlib import Path
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter


# def run_validation(
#     model, validation_ds, tokenizer_src, tokenizer_tgt, max_len,
#     device, print_msg, global_state, writer, num_examples=2
# ):
#     model.eval()
#     count = 0
#     source_texts = []
#     expected = []
#     predicted = []

#     # Size of the control window (default value)
#     console_width = 80

#     with torch.no_grad():
        

def get_ds(config):
    ds_raw = load_dataset(
        'opus_books',
        f'{config["lang_src"]}-{config["lang_tgt"]}', split='train[:1%]'
    )

    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(
        config, ds_raw, config['lang_src']
    )
    tokenizer_tgt = get_or_build_tokenizer(
        config, ds_raw, config['lang_tgt']
    )

    # Keep 90% for train, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, valid_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size]
    )

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )
    valid_ds = BilingualDataset(
        valid_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config['lang_src'],
        config['lang_tgt'],
        config['seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]
        ).ids
        tgt_ids = tokenizer_src.encode(
            item['translation'][config['lang_tgt']]
        ).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True
    )
    val_dataloader = DataLoader(
        valid_ds, batch_size=1, shuffle=True
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config: dict, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config['seq_len'],
        config['seq_len'],
        config['d_model']
    )
    return model


def train_model(config: dict):
    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # TODO continue from there
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config
    )
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
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    
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

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
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
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
