from sagemaker.workflow.steps import (
    TrainingStep, CacheConfig
)
from sagemaker.processing import FrameworkProcessor
from sagemaker.inputs import TrainingInput
from pathlib import Path
from sagemaker.estimator import Estimator
from sagemaker.pytorch import PyTorch

BASE_PATH = Path(__file__).resolve().parent


def create_training_step(
    session: str,
    role: str,
    s3_bucket_name: str,
    instance_type: str,
    instance_count: int,
    train_data,
    tokenizers_path
):
    output_path = f's3://{s3_bucket_name}/models/estimator-models'
    
    estimator = Estimator(
        entry_point='entrypoint.py',
        source_dir=str(BASE_PATH),
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        image_uri='763104351884.dkr.ecr.eu-north-1.amazonaws.com/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker',
        output_path=output_path,
        sagemaker_session=session
    )

    train_input = TrainingInput(
        s3_data=train_data
    )
    tokenizers_path = TrainingInput(
        s3_data=tokenizers_path
    )
    step_train = TrainingStep(
        name='TrainingStep',
        estimator=estimator,
        inputs={
            'training': train_input,
            'tokenizers': tokenizers_path
        },
        cache_config=CacheConfig(
            enable_caching=True,
            expire_after='10d'
        )
    )

    return step_train
