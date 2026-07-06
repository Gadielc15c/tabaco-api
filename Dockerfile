# TFM tobacco classifier - cloud inference server (CPU)
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small; override for GPU hosts if needed.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model_best.pt .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
