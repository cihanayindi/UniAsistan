# 1. Adım: Python'un resmi, hafif bir sürümünü temel al.
FROM python:3.11-slim

# 2. Adım: Çalışma ortamı olarak /code klasörünü ayarla.
WORKDIR /code

# 3. Adım: Önce sadece bağımlılıkları yükle (Docker katman önbelleklemesi için).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. Adım: Model indirme script'ini kopyala ve modeli İMAJIN İÇİNE indir.
# Bu sayede uygulama her başladığında modeli tekrar indirmez.
COPY download_model.py .
RUN python download_model.py

# 5. Adım: Projenin geri kalan tüm dosyalarını /code klasörüne kopyala.
# .dockerignore dosyası kullanarak gereksiz dosyaların kopyalanmasını engelleyebilirsin.
COPY . .

# 6. Adım: Uygulamanın çalışacağı port'u belirt.
EXPOSE 8080

# 7. Adım: Uygulamayı başlatacak olan başlangıç script'ini çalıştır.
# Bu script, hem teşhis bilgisi sağlar hem de uygulamayı başlatır.
CMD ["sh", "start.sh"]