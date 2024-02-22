from pathlib import Path
import tensorflow as tf
import json
import pathlib
from pathlib import Path
from bert_score import BERTScorer
import argparse
import boto3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bucket-name', type=str, help='Bucket name'
    )
    args = parser.parse_args()
    base_dir = '/opt/ml/processing'
    data_path = Path(base_dir) / 'valid/valid_text.txt'
    text_valid = open(str(data_path), 'rb').read().decode(
        encoding='utf-8'
    )
    
    local_model_path = Path(base_dir) / 'model/one_step_model.keras'
    one_step_model = tf.keras.models.load_model(local_model_path)

    idx_start = 8
    length_to_pred = 200
    states = None
    next_char = tf.constant([text_valid[:idx_start]])
    result = [next_char]

    for n in range(length_to_pred):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    pred_valid = tf.strings.join(result)[0].numpy().decode('utf-8')
    true_valid = text_valid[:length_to_pred+8]

    # BERTScore calculation
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([pred_valid], [true_valid])
    print(f'BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}')

    report_dict = {
        'BERTScore_precision': round(P.mean(), 4),
        'BERTScore_recall': round(R.mean(), 4),
        'BERTScore_F1': round(F1.mean(), 4)
    }

    output_dir = Path(base_dir) / "evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    # Send model to S3
    destination_path = 'models/estimator_models'
    s3_client = boto3.client('s3')
    s3_client.upload_file(
        local_model_path, args.bucket_name, destination_path
    )
