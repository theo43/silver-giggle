from sagemaker.processing import (
    ProcessingInput, ProcessingOutput
)
from sagemaker.workflow.parameters import (
    ParameterString
)
from sagemaker.workflow.steps import (
    ProcessingStep, CacheConfig
)
from sagemaker.sklearn.processing import SKLearnProcessor
from pathlib import Path


def create_data_processing_step(
    session: str,
    role: str,
    s3_bucket_name: str,
    instance_count: int,
):
    s3_data_uri = f's3://{s3_bucket_name}/datasets/shakespeare/shakespeare.txt'
    param_input_data = ParameterString(
        name="InputDataShakespeare",
        default_value=s3_data_uri,
    )
    processor_data = SKLearnProcessor(
        framework_version='0.23-1',
        instance_type='ml.t3.medium',
        instance_count=instance_count,
        base_job_name='data-processing-step',
        role=role,
        sagemaker_session=session,
    )

    base_path = Path(__file__).resolve().parent
    entrypoint_path = base_path / 'entrypoint.py'

    step_data_process = ProcessingStep(
        name='DataProcessing',
        processor=processor_data,
        inputs=[
            ProcessingInput(
                source=param_input_data,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='train',
                source='/opt/ml/processing/train'
            ),
            ProcessingOutput(
                output_name='valid',
                source='/opt/ml/processing/valid'
            )
        ],
        code=str(entrypoint_path),
        cache_config=CacheConfig(
            enable_caching=True,
            expire_after='10d'
        )
    )

    return step_data_process, param_input_data
