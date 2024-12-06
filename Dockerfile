# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Streamlit port (if Streamlit is used) or Flask/Django port
EXPOSE 5000

# Command to run your application (adjust based on your framework, e.g., Flask or Streamlit)
CMD ["python", "app.py"]
