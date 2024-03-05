from sagemaker.workflow.steps import TrainingStep, CacheConfig
from sagemaker.inputs import TrainingInput
from pathlib import Path
from sagemaker.estimator import Estimator

BASE_PATH = Path(__file__).resolve().parent


def create_training_step(
    session: str,
    role: str,
    s3_bucket_name: str,
    image_uri_train: str,
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
        instance_type='g5.4xlarge',
        image_uri=image_uri_train,
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
