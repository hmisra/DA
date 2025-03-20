FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama if needed
# Uncomment this if you want to run Ollama inside the container
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user to run the application
RUN useradd -m appuser
USER appuser

# Expose the port the app will run on
EXPOSE 8000

# Set Python path to include source directory
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "src/main.py"] 