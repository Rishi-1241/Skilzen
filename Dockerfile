# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables to avoid writing .pyc files and buffering issues
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY . /app

# Copy the .env file for environment configuration
COPY .env /app/

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable for Google credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials_skillzen.json"

# Run the application
CMD ["python", "app.py"]
