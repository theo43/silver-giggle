# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory
WORKDIR /opt/ml/

# Create the packages folder
RUN mkdir -p ./packages

# Install Python dependencies
RUN pip3 install sagemaker-training
COPY requirements_train.txt ./requirements_train.txt

# Copy all files from local packages folder to /opt/ml/packages
COPY packages/ ./packages/

RUN pip3 install -r ./requirements_train.txt

# # Set the entry point to your training script
# ENV SAGEMAKER_PROGRAM entry_point.py
