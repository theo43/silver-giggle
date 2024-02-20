from pathlib import Path
import pickle


if __name__ == '__main__':
    base_dir = '/opt/ml/processing'
    data_path = Path(base_dir) / 'input/shakespeare.txt'
    text = open(str(data_path), 'rb').read().decode(
        encoding='utf-8'
    )
    # Split between train and validation sets
    split_index = 1003857  # Approx 90% ratio between train/validation
    text_train = text[:split_index]
    text_valid = text[split_index:]

    vocab = sorted(set(text))

    train_path = f'{base_dir}/train/train_text.txt'
    with open(train_path, 'w') as f:
        f.write(text_train)
    
    valid_path = f'{base_dir}/valid/valid_text.txt'
    with open(valid_path, 'w') as f:
        f.write(text_valid)
    
    vocab_path_train = f'{base_dir}/train/vocab.pkl'
    with open(vocab_path_train, 'wb') as f:
        pickle.dump(vocab, f)
