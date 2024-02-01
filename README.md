# silver-giggle

[![Training](https://github.com/theo43/silver-giggle/actions/workflows/training_workflow.yml/badge.svg)](https://github.com/theo43/silver-giggle/actions/workflows/training_workflow.yml)

This project aims at using MLOps best practices to create, validate NLP models on AWS SageMaker, and productionize them with an API. 

## Architecture
This single repository contains everything you need to `experiment` (train and validate
your models), and then create the Docker images you need to push these models to `production`. `packages`
is where the code used on one hand to validate models during experiments, on the other hand in production,
is tested and packaged. It is then only written once at a single place to avoid duplications.

## Experimenting
### With a local Jupyter Lab
Activate a Python 3.11 installation, then:
```
cd experiment/notebooks
python -m venv venv-jup
source ./venv-jup/bin/activate
pip install -r requirements.txt
```

To link your venv to a Jupyter notebook kernel, run with your activated `venv-jup`:
```
python -m ipykernel install --user --name=venv-jup
```
You can then use that env in an activated Jupyter notebook. `/experiment/notebooks` contains notebooks for training/validating NLP models.

### With a local Sagemaker
```
cd experiment/train
docker build -t <train_image_name> .
docker tag <train_image_name> <github_username>/<train_image_name>:latest
docker login
docker push <github_username>/<train_image_name>:latest
python pipeline_train.py --s3-bucket-name <bucket_name> --role <role> --image-uri <github_username>/<train_image_name>:latest
```
Today it seems that local SageMaker training is impossible with custom Docker image since the docker-compose.yaml file generated on the run has a bad shared memory size (`shm_size`) value by default.

### On AWS SageMaker
Setup an AWS account, create a user, a role that can be assumed by the user and provide those credentials to GithubActions secrets for CI/CD.

## Packages
TODO: describe how to test and build packages locally

## Production
TODO: describe how to test, build and deploy the API locally

## Sources
This repository is inspired by the following sources:
- [KREUZBERGER, Dominik, KÜHL, Niklas, et HIRSCHL, Sebastian. Machine learning operations (mlops): Overview, definition, and architecture. IEEE Access, 2023.](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10081336)
- [Tensorflow tutorial for text generation using an RNN](https://www.tensorflow.org/text/tutorials/text_generation)
- [Tensorflow tutorial for text translation using Transformers](https://www.tensorflow.org/text/tutorials/transformer)
