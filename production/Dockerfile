# Use the official Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY ./production/requirements.txt .
COPY ./production/translation-0.0.0-py3-none-any.whl .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and model files to the working directory
COPY ./production/* .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]
