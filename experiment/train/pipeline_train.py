from sagemaker.workflow.pipeline_context import LocalPipelineSession
from sagemaker.tensorflow import TensorFlow
import sagemaker
from sagemaker.workflow.steps import TrainingStep, TrainingInput
from sagemaker.workflow.pipeline import Pipeline
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Text generation model training'
    )
    parser.add_argument(
        '--s3-bucket-name', type=str, help='AWS S3 bucket name'
    )

    args = parser.parse_args()
    s3_bucket_name = args.s3_bucket_name

    session = LocalPipelineSession()
    #role = sagemaker.get_execution_role()
    role = 'sagemaker-silver-role'

    tensorflow_estimator = TensorFlow(
        sagemaker_session=session,
        role=role,
        instance_type='ml.c5.xlarge',
        instance_count=1,
        framework_version='2.13',
        py_version='py310',
        entry_point='entry_point_train.py'
    )

    s3_train_data = f's3://{s3_bucket_name}/datasets/shakespeare/shakespeare.txt'

    step = TrainingStep(
        name="Shakespeare training step",
        step_args=tensorflow_estimator.fit(
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
