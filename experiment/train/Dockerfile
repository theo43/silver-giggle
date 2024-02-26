# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory
WORKDIR /opt/ml/code

# Install Python dependencies
RUN pip3 install sagemaker-training
COPY requirements_train.txt /opt/ml/code/requirements_train.txt
COPY packages /opt/ml/code/packages
RUN pip3 install -r requirements_train.txt

# # Set the entry point to your training script
# ENV SAGEMAKER_PROGRAM entry_point.py
