# Python'un stabil bir sürümünü temel al
FROM python:3.11-slim

# Çalışma dizinini /app olarak ayarla
WORKDIR /app

# Önce sadece requirements.txt dosyasını kopyala
# Bu, Docker'ın katman önbelleğini verimli kullanmasını sağlar
COPY requirements.txt .

# Gerekli kütüphaneleri kur
RUN pip install --no-cache-dir -r requirements.txt

# Geri kalan tüm proje dosyalarını kopyala
COPY . .

# Uygulamayı başlatacak komutu belirt
# Fly.io bu porta otomatik olarak bağlanacaktır
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]