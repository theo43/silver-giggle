from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession, PipelineSession
)
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline import Pipeline
# from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker import get_execution_role
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Text generation model training parser'
    )
    parser.add_argument(
        '--s3-bucket-name', type=str, help='AWS S3 bucket name'
    )
    parser.add_argument(
        '--image-uri', type=str, help='Training image URI'
    )
    parser.add_argument(
        '--local', type=bool, default=False,
        help='Local mode execution'
    )
    parser.add_argument(
        '--subnet-id1', type=str, help='Subnet id1'
    )
    parser.add_argument(
        '--subnet-id2', type=str, help='Subnet id2'
    )
    parser.add_argument(
        '--subnet-id3', type=str, help='Subnet id3'
    )
    parser.add_argument(
        '--security-group-id', type=str, help='Security group id'
    )

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name
    role = get_execution_role()
    image_uri = args.image_uri
    local = args.local


    if local:
        session = LocalPipelineSession()
        session.config = {'local': {'local_code': True}}
        instance_count = 1
        instance_type = 'local'
    else:
        session = PipelineSession()
        instance_count = 1
        instance_type = 'ml.m5.large'

    output_path = f's3://{s3_bucket_name}/models/estimator-models'
    # estimator = Estimator(
    #    image_uri=image_uri,
    #    role=role,
    #    instance_type=instance_type,
    #    instance_count=instance_count,
    #    source_dir='src',
    #    entry_point='entry_point_train.py',
    #    output_path=output_path,
    #    training_repository_access_mode='Vpc',
    #    subnets=[
    #        args.subnet_id1, args.subnet_id2, args.subnet_id3
    #    ],
    #    security_group_ids=[args.security_group_id]
    # )
    estimator = TensorFlow(
        py_version='py310',
        framework_version='2.13',
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        source_dir='src',
        entry_point='entry_point_train.py',
        output_path=output_path
    )

    s3_train_data = f's3://{s3_bucket_name}/datasets/shakespeare/shakespeare.txt'

    step = TrainingStep(
        name="Shakespeare training step",
        step_args=estimator.fit({'training': s3_train_data})
    )

    pipeline = Pipeline(
        name='Training pipeline',
        steps=[step],
        sagemaker_session=session
    )

    pipeline.upsert(
        role_arn=role,
        description='Shakespeare training pipeline'
    )

    execution = pipeline.start()
