# Use a specific, stable Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables for Python
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install essential system packages that some Python libraries might need
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to use Docker's caching mechanism
COPY requirements.txt .

# Install all the Python libraries from your requirements file
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# The hosting platform will use the Procfile to run the commands,
# so no CMD or ENTRYPOINT is needed here.