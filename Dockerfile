FROM python:3.9-slim

# Invalidate cache with a temporary label
LABEL cache-buster="2026-02-12-v1"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --force-reinstall -r requirements.txt

COPY . .

CMD ["python", "train.py"]