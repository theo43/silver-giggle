from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.tensorflow import TensorFlow
import sagemaker
from sagemaker.workflow.steps import TrainingStep, TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.estimator import Estimator
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Text generation model training'
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

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name
    role = args.role
    image_uri = args.image_uri

    session = LocalPipelineSession()
    session.config = {'local': {'local_code': True}}
    instance_count = 1
    instance_type = 'local'

    estimator = Estimator(
       image_uri=image_uri,
       role=role,
       instance_type=instance_type,
       instance_count=instance_count,
       output_path='',
    )
    estimator.set_hyperparameters({'shm_size': '1G'})

    s3_train_data = f's3://{s3_bucket_name}/datasets/shakespeare/shakespeare.txt'

    step = TrainingStep(
        name="Shakespeare training step",
        step_args=estimator.fit(
            inputs=TrainingInput(s3_data=s3_train_data)
            )
    )

    pipeline = Pipeline(
        name='Training pipeline',
        steps=[step],
        sagemaker_session=session
    )

    pipeline.upsert(
        role_arn=role,
        description='local pipeline'
    )

    execution = pipeline.start()
