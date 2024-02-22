# Use the official Ubuntu base image with Python 3.11
FROM tensorflow/tensorflow:2.15.0

# Set the working directory
WORKDIR /opt/ml/code

# Install Python dependencies
RUN pip3 install sagemaker-training
COPY ./experiment/train/requirements-train.txt .
RUN pip3 install --no-cache-dir -r ./requirements-train.txt

#ENV MLFLOW_TRACKING_URI=to_be_replaced_by_ci_cd

# # Copy training script
# COPY entry_point_train.py .

# # Set the entry point to your training script
# ENV SAGEMAKER_PROGRAM entry_point.py
