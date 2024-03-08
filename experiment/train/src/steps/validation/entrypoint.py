import torch
from pathlib import Path
from datasets import load_dataset
from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate
from tokenizers import Tokenizer
from translation.config import get_config
from translation.model import get_model
from translation.dataset import causal_mask


def greedy_decode(
    model, source, source_mask, tokenizer_src, tokenizer_tgt,
    max_len, device
):
    sos_id = tokenizer_tgt.token_to_id('[SOS]')
    eos_id = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it
    # for every token we get from decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder input with SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_id).type_as(source).to(device)
    while True:
        if decoder_input.size(1) > max_len:
            break

        # Build mask for the decoder input
        decoder_mask = causal_mask(
            decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculate output of decoder
        out = model.decode(
            encoder_output, source_mask, decoder_input, decoder_mask
        )

        # Get next token
        prob = model.project(out[:, -1])
        # Select the token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == eos_id:
            break
    
    return decoder_input.squeeze(0)


if __name__ == '__main__':
    processing_dir = '/opt/ml/processing'

    # Create config
    config = get_config()
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    ds_raw = load_dataset(
        'opus_books',
        f'{lang_src}-{lang_tgt}',
        split=f'train[:{config["download_size"]}%]'
    )

    # Load valid dataloader, tokenizers and model weights
    model_checkpoint = torch.load(
        str(Path(processing_dir) / '/model/weights/tmodel_02.pt'),
        map_location=device
    )
    valid_dataloader = torch.load(
        str(Path(processing_dir) / '/valid/valid_dataloader.pkl')
    )
    tokenizer_src = Tokenizer.from_file(
        str(Path(processing_dir) / f'/tokenizers/tokenizer_{lang_src}.json')
    )
    tokenizer_tgt = Tokenizer.from_file(
        str(Path(processing_dir) / f'/tokenizers/tokenizer_{lang_tgt}.json')
    )

    # Create model, load previously trained weights and set to eval mode
    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size()
    ).to(device)
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()

    count = 0
    source_texts = []
    expected = []
    predicted = []

    with torch.no_grad():
        for batch in valid_dataloader:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            assert encoder_input.size(0) == 1, 'Batch size must be 1'

            model_output = greedy_decode(
                model, encoder_input, encoder_mask, tokenizer_src,
                tokenizer_tgt, config['seq_len'], device
            )

            source_txt = batch['src_text'][0]
            target_txt = batch['tgt_text'][0]

            model_out_txt = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy()
            )

            source_texts.append(source_txt)
            expected.append(target_txt)
            predicted.append(model_out_txt)

            print(f'SOURCE: {source_txt}')
            print(f'TARGET: {target_txt}')
            print(f'PREDICTED: {model_out_txt}')
    
    # Compute BLEU score
    metric = BLEUScore()
    bleu_score = metric(predicted, expected)
    print(f'BLEU score: {bleu_score.numpy().tolist():.3f}')
    
    # Compute Char Error Rate
    metric = CharErrorRate()
    cer = metric(predicted, expected)
    print(f'Char Error Rate: {cer.numpy().tolist():.3f}')

    # Compute Word Error Rate
    metric = WordErrorRate()
    wer = metric(predicted, expected)
    print(f'Word Error Rate: {wer.numpy().tolist():.3f}')
