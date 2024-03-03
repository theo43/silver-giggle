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
        '--local', type=bool, default=False,
        help='Local mode execution'
    )

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name
    local = args.local


    if local:
        session = LocalPipelineSession()
        # session.config = {'local': {'local_code': True}}
        role = args.role
        instance_count = 1
        instance_type = 'local'
    else:
        session = PipelineSession()
        role = get_execution_role()
        instance_count = 1
        instance_type = 'ml.m5.2xlarge'

    # Data processing step
    step_data_process = create_data_processing_step(
        session,
        role,
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
        instance_type,
        instance_count,
        train_data,
        tokenizers_path
    )

    # Evaluation step
    # TODO

    # Define the whole pipeline
    pipeline = Pipeline(
        name='MachineTranslationPipeline3',
        # parameters=[
        #     param_input_data
        # ],
        steps=[
            step_data_process,
            step_train,
            # step_eval
        ],
        sagemaker_session=session
    )
    pipeline.upsert(
        role_arn=role,
        description='translation-pipeline'
    )
    execution = pipeline.start()
