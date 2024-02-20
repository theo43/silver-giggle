# processor_eval = ScriptProcessor(
    #     image_uri=image_uri,
    #     command=['python3'],
    #     instance_type='ml.t3.medium',
    #     instance_count=1,
    #     base_job_name='bert-score-evaluation',
    #     role=role
    # )
    # evaluation_report = PropertyFile(
    #     name='EvaluationReport',
    #     output_name='evaluation',
    #     path='evaluation.json'
    # )
    # step_eval = ProcessingStep(
    #     name='EvaluationStep',
    #     processor=processor_eval,
    #     inputs=[
    #         ProcessingInput(
    #             source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #             destination='/opt/ml/processing/model'
    #         ),
    #         ProcessingInput(
    #             source=step_data_process.properties.ProcessingOutputConfig.Outputs[
    #                 'valid'
    #             ].S3Output.S3Uri,
    #             destination='/opt/ml/processing/valid'
    #         )
    #     ],
    #     outputs=[
    #         ProcessingOutput(
    #             output_name='evaluation',
    #             source='/opt/ml/processing/evaluation'
    #         )
    #     ],
    #     code='./src/entrypoint_evaluation.py',
    #     property_files=[evaluation_report],
    #     job_arguments=[
    #         '--bucket-name', s3_bucket_name,
    #     ]
    # )