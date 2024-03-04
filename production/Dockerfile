# Use the official Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY ./production/requirements_production.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements_production.txt

# Copy the code and model files to the working directory
COPY ./production/* .

# Expose the port on which the FastAPI application will run
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
