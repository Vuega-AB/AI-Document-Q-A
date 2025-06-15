# Use a specific, stable Python version
FROM python:3.13.3

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install essential system packages, including dos2unix
RUN apt-get update && apt-get install -y \
    build-essential \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to use Docker's caching mechanism
COPY requirements.txt .

# Install all the Python libraries from your requirements file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# IMPORTANT: Ensure the start script has correct permissions and line endings
RUN dos2unix ./start.sh
RUN chmod +x ./start.sh

# The hosting platform will use the start.sh script as the start command.