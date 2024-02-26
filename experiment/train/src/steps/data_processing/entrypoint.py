from pathlib import Path
import pickle
from torch.utils.data import DataLoader, random_split
from translation.dataset import BilingualDataset
from translation.tokenizer import get_or_build_tokenizer
from translation.config import get_config


if __name__ == '__main__':
    config = get_config()
    lang_src = config['lang_src']
    lang_tgt = config['lang_tgt']
    base_dir = '/opt/ml/processing'
    data_path = Path(base_dir) / 'input/en-es_dataset.pickle'
    with open(data_path, 'rb') as data:
        ds_raw = pickle.load(data)
    
    # Build tokenizer
    tokenizer_src = get_or_build_tokenizer(
        config, ds_raw, lang_src
    )
    tokenizer_tgt = get_or_build_tokenizer(
        config, ds_raw, lang_tgt
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
        lang_src,
        lang_tgt,
        config['seq_len']
    )
    valid_ds = BilingualDataset(
        valid_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        lang_src,
        lang_tgt,
        config['seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][lang_src]
        ).ids
        tgt_ids = tokenizer_src.encode(
            item['translation'][lang_tgt]
        ).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_ds, batch_size=1, shuffle=True
    )

    train_path = f'{base_dir}/train'
    with open(str(Path(train_path) / 'train_dataloader.pickle'), 'wb') as f:
        pickle.dump(train_dataloader, f)
    
    valid_path = f'{base_dir}/valid'
    with open(str(Path(valid_path) / 'valid_dataloader.pickle'), 'wb') as f:
        pickle.dump(valid_dataloader, f)
    
    tokenizer_src.save(str(Path(train_path) / f'tokenizer_{lang_src}].json'))
    tokenizer_src.save(str(Path(valid_path) / f'tokenizer_{lang_src}.json'))
    tokenizer_tgt.save(str(Path(train_path) / f'tokenizer_{lang_tgt}.json'))
    tokenizer_tgt.save(str(Path(valid_path) / f'tokenizer_{lang_tgt}.json'))
