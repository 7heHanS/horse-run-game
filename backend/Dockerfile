FROM python:3.11-slim

WORKDIR /app

# Install dependencies (Minimal for log collection)
RUN pip install fastapi uvicorn pydantic --no-cache-dir

# Copy the backend source
COPY . /app

# Expose the API port
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
