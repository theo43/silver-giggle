from sagemaker.processing import (
    ProcessingOutput, ProcessingInput, FrameworkProcessor
)
from sagemaker.workflow.steps import (
    ProcessingStep, CacheConfig
)
from sagemaker.pytorch import PyTorch
from pathlib import Path


def create_validation_step(
    session: str,
    role: str,
    valid_data,
    model_path,
    **kwargs
):
    base_path = Path(__file__).resolve().parent
    processing_path = '/opt/ml/processing'

    processor = FrameworkProcessor(
        estimator_cls=PyTorch,
        framework_version='2.2',
        image_uri=kwargs['image_uri'],
        instance_type=kwargs['instance_type'],
        instance_count=kwargs['instance_count'],
        base_job_name='data-processing-step',
        role=role,
        sagemaker_session=session,
    )

    # evaluation_report = PropertyFile(
    #     name='EvaluationReport',
    #     output_name='evaluation',
    #     path='evaluation.json'
    # )

    step_args = processor.run(
        inputs=[
            ProcessingInput(
                input_name='validation-data',
                source=valid_data,
                destination=f'{processing_path}/valid'
            ),
            ProcessingInput(
                input_name='model',
                source=model_path,
                destination=f'{processing_path}/model'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name='evaluation',
                source=f'{processing_path}/evaluation'
            )
        ],
        code='entrypoint.py',
        source_dir=str(base_path),
    )

    step_validation = ProcessingStep(
        name='ValidationStep',
        step_args=step_args,
        cache_config=CacheConfig(
            enable_caching=False,
            expire_after='10d'
        )
    )

    return step_validation
