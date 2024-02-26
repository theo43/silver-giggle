from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession, PipelineSession
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker import get_execution_role
import argparse
from src.steps.data_processing.step_creator import (
    create_data_processing_step
)
from src.steps.training.step_creator import (
    create_training_step
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
        '--image-uri', type=str, help='Training image URI'
    )
    parser.add_argument(
        '--image-ecr-uri', type=str, help='Training image ECR URI'
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
    image_uri = args.image_uri
    local = args.local


    if local:
        session = LocalPipelineSession()
        session.config = {'local': {'local_code': True}}
        role = args.role
        instance_count = 1
        instance_type = 'local'
    else:
        session = PipelineSession()
        role = get_execution_role()
        instance_count = 1
        instance_type = 'ml.m5.large'

    # Data processing step
    step_data_process, param_input_data = create_data_processing_step(
        session,
        role,
        s3_bucket_name,
        instance_count
    )

    # Training step
    train_data = step_data_process.properties.ProcessingOutputConfig.Outputs[
        'train'
    ].S3Output.S3Uri

    step_train = create_training_step(
        session,
        role,
        s3_bucket_name,
        instance_type,
        instance_count,
        train_data
    )

    # Evaluation step
    # TODO

    # Define the whole pipeline
    pipeline = Pipeline(
        name='ShakespearePipeline',
        parameters=[
            param_input_data
        ],
        steps=[
            step_data_process,
            step_train,
            # step_eval
        ],
        sagemaker_session=session
    )
    pipeline.upsert(
        role_arn=role,
        description='shakespeare-pipeline'
    )
    execution = pipeline.start()
