from sagemaker.processing import (
    ProcessingOutput, FrameworkProcessor
)
from sagemaker.workflow.steps import (
    ProcessingStep, CacheConfig
)
from sagemaker.pytorch import PyTorch
from pathlib import Path


def create_data_processing_step(
    session: str,
    role: str,
    image_uri_valid: str,
    instance_type: str,
    instance_count: int,
):
    base_path = Path(__file__).resolve().parent
    processing_path = '/opt/ml/processing'
    
    processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version='2.2',
        image_uri=image_uri_valid,
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name='data-processing-step',
        role=role,
        sagemaker_session=session,
    )

    step_args = processor.run(
        outputs=[
            ProcessingOutput(
                output_name='train_data',
                source=f'{processing_path}/output/train/'
            ),
            ProcessingOutput(
                output_name='valid_data',
                source=f'{processing_path}/output/valid/'
            ),
            ProcessingOutput(
                output_name='tokenizers',
                source=f'{processing_path}/output/tokenizers/'
            )
        ],
        code='entrypoint.py',
        source_dir=str(base_path),
    )

    step_data_process = ProcessingStep(
        name='DataProcessing',
        step_args=step_args,
        cache_config=CacheConfig(
            enable_caching=True,
            expire_after='10d'
        )
    )

    return step_data_process
