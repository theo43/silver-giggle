from sagemaker.workflow.steps import (
    TrainingStep, CacheConfig
)
from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from pathlib import Path


def create_training_step(
    session,
    role: str,
    s3_bucket_name: str,
    instance_type: str,
    instance_count: int,
    train_data
):
    output_path = f's3://{s3_bucket_name}/models/estimator-models'

    estimator = TensorFlow(
        py_version='py310',
        framework_version='2.13',
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        source_dir=str(Path(__file__).resolve().parent),
        entry_point='entrypoint.py',
        output_path=output_path
    )
    train_input = TrainingInput(
        s3_data=train_data
    )
    step_train = TrainingStep(
        name='TrainingStep',
        estimator=estimator,
        inputs={'training': train_input},
        cache_config=CacheConfig(
            enable_caching=True,
            expire_after='10d'
        )
    )

    return step_train
