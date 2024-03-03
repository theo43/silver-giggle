from sagemaker.processing import (
    ProcessingInput, ProcessingOutput, FrameworkProcessor
)
from sagemaker.workflow.steps import (
    ProcessingStep, CacheConfig
)
from sagemaker.pytorch import PyTorch
from pathlib import Path


def create_data_processing_step(
    session: str,
    role: str,
    s3_bucket_name: str,
    instance_count: int,
):
    base_path = Path(__file__).resolve().parent
    # entrypoint_path = base_path / 'entrypoint.py'
    processing_path = '/opt/ml/processing'

    s3_data_uri = f's3://{s3_bucket_name}/datasets/translation/en-es_dataset.pickle'
    
    processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version='2.2',
        image_uri='763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker',
        instance_type='ml.t3.medium',
        instance_count=instance_count,
        base_job_name='data-processing-step',
        role=role,
        # sagemaker_session=session,
    )

    step_args = processor.run(
        inputs=[
            ProcessingInput(
                input_name='input_data',
                source=s3_data_uri,
                destination=f'{processing_path}/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train_data',
                source=f'{processing_path}/output/train/train_dataloader.pickle',
                # destination=f's3://{s3_bucket_name}/datasets/translation/processed/train/train_dataloader.pickle'
            ),
            ProcessingOutput(
                output_name='valid_data',
                source=f'{processing_path}/output/valid/valid_dataloader.pickle',
                # destination=f's3://{s3_bucket_name}/datasets/translation/processed/valid/valid_dataloader.pickle'
            ),
            ProcessingOutput(
                output_name='tokenizers',
                source=f'{processing_path}/output/tokenizers/',
                # destination=f's3://{s3_bucket_name}/datasets/translation/processed/tokenizers/'
            )
        ],
        code='entrypoint.py',
        source_dir=str(base_path),
    )

    step_data_process = ProcessingStep(
        name='DataProcessing',
        step_args=step_args,
        cache_config=CacheConfig(
            enable_caching=False,
            expire_after='10d'
        )
    )

    return step_data_process
