# Gunakan base image Python yang stabil dan ringan untuk Machine Learning
FROM python:3.11-slim

# Variabel environment Python dan Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install dependensi sistem yang dibutuhkan oleh LightGBM (libgomp1) dan healthcheck (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt terlebih dahulu untuk memanfaatkan Docker cache layer
COPY requirements.txt .

# Install paket Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi, model, dan dataset
COPY app.py .
COPY model/ ./model/
COPY data/ ./data/
COPY exported/ ./exported/

# Buka port default Streamlit
EXPOSE 8501

# Healthcheck untuk memonitor status container
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Jalankan aplikasi Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
