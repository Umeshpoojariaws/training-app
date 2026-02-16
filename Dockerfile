FROM python:3.9-slim

WORKDIR /app

# First, copy only the requirements file to leverage Docker layer caching.
# If requirements.txt doesn't change, this layer won't be re-run.
COPY requirements.txt .

# Upgrade pip and install all dependencies in a single layer.
# --no-cache-dir is used to keep the image size smaller.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code.
COPY . .

CMD ["python", "train.py"]