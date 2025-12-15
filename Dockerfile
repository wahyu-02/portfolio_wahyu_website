# Gunakan image Python ringan
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode
COPY . .

# Jalankan aplikasi dengan Gunicorn
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
