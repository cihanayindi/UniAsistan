# Stage 1: Bağımlılıkları kurmak için builder aşaması
FROM python:3.11-slim as builder

WORKDIR /app

# Pip'i en son sürüme güncelle
RUN pip install --upgrade pip

# Sanal ortam yerine, bağımlılıkları belirli bir dizine kur
ENV PIP_TARGET=/app/packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target=$PIP_TARGET

# ----------------------------------------------------------------

# Stage 2: Asıl çalışma imajı
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/packages

# Builder aşamasından sadece kurulan paketleri kopyala
COPY --from=builder /app/packages /app/packages

# --------> EN ÖNEMLİ SATIR <--------
# Lokal'de oluşturduğumuz hazır veritabanını imajın içine kopyala
COPY data/ /app/data/

# Uygulamanın geri kalanını kopyala (main.py vs.)
COPY . .

# Uygulamanın çalışacağı port
EXPOSE 8080

# Uygulamayı çalıştır
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]