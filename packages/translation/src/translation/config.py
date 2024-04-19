from pathlib import Path


def get_config():
    return {
        'batch_size': 8,
        'num_epochs': 3,
        'download_size': 100,
        'lr': 10**-4,
        'seq_len': 350,
        'd_model': 512,
        'lang_src': 'en',
        'lang_tgt': 'es',
        'model_folder': 'weights',
        'model_filename': 'tmodel_',
        'preload': None,
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
        'seed': 34
    }


def get_weights_file_path(config, epoch: str):
    model_filename = f'{config["model_filename"]}{epoch}.pt'
    model_filepath = Path('.') / config['model_folder'] / model_filename
    return str(model_filepath)
