# silver-giggle

[![Training](https://github.com/theo43/silver-giggle/actions/workflows/training_workflow.yml/badge.svg)](https://github.com/theo43/silver-giggle/actions/workflows/training_workflow.yml)
[![Production](https://github.com/theo43/silver-giggle/actions/workflows/production_workflow.yml/badge.svg)](https://github.com/theo43/silver-giggle/actions/workflows/production_workflow.yml)

This project aims at using MLOps best practices to create, validate NLP models on the cloud, and productionize them with an API. 

## Architecture
This single repository contains everything you need to `experiment` (train and validate
your models), and then create the Docker images you need to push these models to `production`. `packages`
is where the code used on one hand to validate models during experiments, on the other hand in production,
is tested and packaged. It is then only written once at a single place to avoid duplications.

## Experimenting
### AWS setup
Setup an AWS account, create a user, a role that can be assumed by the user and provide those credentials to GithubActions secrets for CI/CD.

### Local Sagemaker
You can verify that your pipeline runs well locally instead of doing it on expensive computes. To do so, with a Python 3.10 installation activated run:
```
cd experiment/train
python -m venv venv
source venv/bin/activate
pip install sagemaker==2.204.0
python pipeline_train.py --role <your_role> --local True
```

### On AWS SageMaker
The run of pipelines on AWS is automated via Github Actions. See `./github/workflows/*`

## Packages
`packages` contains the code used to validate models during experiments and in production. It is tested and packaged to avoid duplications.

```
cd packages/<package_name>/src
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python -m unittest discover

```

## Production
TODO: describe how to test, build and deploy the API locally

## Sources
This repository is inspired by the following sources:
- [KREUZBERGER, Dominik, KÜHL, Niklas, et HIRSCHL, Sebastian. Machine learning operations (mlops): Overview, definition, and architecture. IEEE Access, 2023.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10081336)
- [Tensorflow tutorial for text generation using an RNN](https://www.tensorflow.org/text/tutorials/text_generation)
- [Tensorflow tutorial for text translation using Transformers](https://www.tensorflow.org/text/tutorials/transformer)
- https://www.youtube.com/watch?v=ISNdQcPhsts
- https://github.com/hkproj/pytorch-transformer

