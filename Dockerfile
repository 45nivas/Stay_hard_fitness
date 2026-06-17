# Use a stable, lightweight Python image
FROM python:3.10-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system-level dependencies required for PostgreSQL, OpenCV, and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Collect static files for production
RUN python manage.py collectstatic --no-input

# Expose port (Render sets this dynamically, but 8000 is default)
EXPOSE 8000

# Run migrations and start Gunicorn binding to Render's dynamic port
CMD ["sh", "-c", "python manage.py migrate && gunicorn gym_project.wsgi:application --bind 0.0.0.0:${PORT:-8000}"]
