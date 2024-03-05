from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession, PipelineSession
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker import get_execution_role
import argparse
import yaml
from pathlib import Path
from src.steps.data_processing.step_creator import (
    create_data_processing_step
)
from src.steps.training.step_creator import (
    create_training_step
)
from src.steps.validation.step_creator import (
    create_validation_step
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Translation model training parser'
    )
    parser.add_argument(
        '--s3-bucket-name', type=str, help='AWS S3 bucket name'
    )
    parser.add_argument(
        '--role', type=str, help='AWS role'
    )
    parser.add_argument(
        '--local', type=bool, default=False,
        help='Local mode execution'
    )

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name
    local = args.local

    aws_config_path = Path(__file__).resolve().parent.joinpath('aws_config.yml')
    aws_config = yaml.safe_load(aws_config_path.read_text())

    if local:
        session = LocalPipelineSession()
        session.config = {'local': {'local_code': True}}
        role = args.role
        for _, v in aws_config.items():
            v['instance_type'] = 'local'
    else:
        session = PipelineSession()
        role = get_execution_role()

    # Data processing step
    step_data_process = create_data_processing_step(
        session,
        role,
        **aws_config['data_processing_step']
    )

    # Training step
    train_data = step_data_process.properties.ProcessingOutputConfig.Outputs[
        'train_data'
    ].S3Output.S3Uri
    tokenizers_path = step_data_process.properties.ProcessingOutputConfig.Outputs[
        'tokenizers'
    ].S3Output.S3Uri

    step_train = create_training_step(
        session,
        role,
        s3_bucket_name,
        train_data,
        tokenizers_path,
        **aws_config['training_step']
    )

    # Evaluation step
    valid_data = step_data_process.properties.ProcessingOutputConfig.Outputs[
        'valid_data'
    ].S3Output.S3Uri
    model_path = step_train.properties.ModelArtifacts.S3ModelArtifacts
    step_validation = create_validation_step(
        session,
        role,
        valid_data,
        model_path,
        tokenizers_path,
        **aws_config['validation_step']
    )

    # Define the whole pipeline
    pipeline = Pipeline(
        name='MachineTranslationPipeline3',
        steps=[
            step_data_process,
            step_train,
            step_validation
        ],
        sagemaker_session=session
    )
    pipeline.upsert(
        role_arn=role,
        description='translation-pipeline'
    )
    execution = pipeline.start()
