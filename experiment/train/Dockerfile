# Use the official Python 3.11 image
FROM python:3.11

# Set the working directory
WORKDIR /opt/ml/

# Create the packages folder
RUN mkdir -p ./code/packages

# Install Python dependencies
RUN pip3 install sagemaker-training
COPY requirements_train.txt ./code/requirements_train.txt

# Copy all files from local packages folder to /opt/ml/packages
COPY packages/ ./code/packages/

RUN pip3 install -r ./code/requirements_train.txt

# # Set the entry point to your training script
# ENV SAGEMAKER_PROGRAM entry_point.py
