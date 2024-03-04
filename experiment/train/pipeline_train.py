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
from src.steps.validation.step_creator import (
    create_validation_step
)

BASE_IMAGE_URL = '763104351884.dkr.ecr.eu-north-1.amazonaws.com'


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
    parser.add_argument(
        '--image-uri-valid', type=str,
        help='Image for validation',
        default=f'{BASE_IMAGE_URL}/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker'
    )
    parser.add_argument(
        '--image-uri-train', type=str,
        help='Image for train',
        default=f'{BASE_IMAGE_URL}/pytorch-training:2.2.0-cpu-py310-ubuntu20.04-sagemaker'
    )

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name
    local = args.local
    image_uri_valid = args.image_uri_valid
    image_uri_train = args.image_uri_train

    if local:
        session = LocalPipelineSession()
        session.config = {'local': {'local_code': True}}
        role = args.role
        instance_count = 1
        instance_type_cpu = 'local'
        instance_type_gpu = 'local'
    else:
        session = PipelineSession()
        role = get_execution_role()
        instance_count = 1
        instance_type_cpu = 'ml.t3.medium'
        instance_type_gpu = 'ml.p3.8xlarge'

    # Data processing step
    step_data_process = create_data_processing_step(
        session,
        role,
        image_uri_valid,
        instance_type_cpu,
        instance_count
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
        image_uri_train,
        instance_type_gpu,
        instance_count,
        train_data,
        tokenizers_path
    )

    # Evaluation step
    valid_data = step_data_process.properties.ProcessingOutputConfig.Outputs[
        'valid_data'
    ].S3Output.S3Uri
    model_path = step_train.properties.ModelArtifacts.S3ModelArtifacts
    step_validation = create_validation_step(
        session,
        role,
        image_uri_valid,
        instance_type_cpu,
        instance_count,
        valid_data,
        model_path,
        tokenizers_path
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
