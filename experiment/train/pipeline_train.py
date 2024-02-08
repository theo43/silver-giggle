from sagemaker.workflow.pipeline_context import (
    LocalPipelineSession, PipelineSession
)
from sagemaker.workflow.steps import (
    TrainingStep, ProcessingStep
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterString, ParameterInteger
)
from sagemaker.processing import (
    ProcessingInput, ProcessingOutput
)
# from sagemaker.estimator import Estimator
from sagemaker.tensorflow import TensorFlow
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
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
        '--role', type=str, help='AWS role'
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
    
    s3_data_uri = f's3://{s3_bucket_name}/datasets/shakespeare/shakespeare.txt'
    param_input_data = ParameterString(
        name="InputDataShakespeare",
        default_value=s3_data_uri,
    )
    processor = SKLearnProcessor(
        framework_version='0.23-1',
        instance_type=instance_type,
        instance_count=instance_count,
        base_job_name='data-processing-process',
        role=role,
        sagemaker_session=session,
    )

    step_data_process = ProcessingStep(
        name='DataProcessing',
        processor=processor,
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
        code='./src/data_processing.py'
    )

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

    # training_input = TrainingInput(s3_data_uri)

    train_input = TrainingInput(
        s3_data=step_data_process.properties.ProcessingOutputConfig.Outputs[
            'train'
        ].S3Output.S3Uri
    )
    step_train = TrainingStep(
        name="TrainingStep",
        estimator=estimator,
        inputs={'training': train_input}
    )

    pipeline = Pipeline(
        name='ShakespearePipeline',
        parameters=[
            param_input_data
        ],
        steps=[
            step_data_process,
            step_train
        ],
        sagemaker_session=session
    )

    pipeline.upsert(
        role_arn=role,
        description='shakespeare-pipeline'
    )

    execution = pipeline.start()
