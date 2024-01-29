# Use the official Ubuntu base image with Python 3.11
FROM tensorflow/tensorflow:2.15.0

# Set the working directory
WORKDIR /opt/ml/code

# Install Python dependencies
COPY ./requirements-exp.txt .
RUN pip install --no-cache-dir -r ./requirements-exp.txt

# Copy training script
COPY entry_point_train.py .

# Set the entry point to your training script
ENTRYPOINT ["python3", "entry_point_train.py"]
