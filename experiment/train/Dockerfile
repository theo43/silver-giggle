# Use the official Ubuntu base image with Python 3.10
FROM ubuntu:latest

# Set the working directory
WORKDIR /opt/ml/code

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Install Python dependencies
COPY ../requirements-exp.txt .
RUN pip install --no-cache-dir -r ./requirements-exp.txt

# Copy training script
COPY entry_point_train.py .

# Set the entry point to your training script
ENTRYPOINT ["python3.10", "entry_point_train.py"]
