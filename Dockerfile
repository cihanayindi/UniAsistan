# Python'un stabil bir sürümünü temel al
FROM python:3.11-slim

# Çalışma dizinini /app olarak ayarla
WORKDIR /app

# Önce sadece requirements.txt dosyasını kopyala
COPY requirements.txt .

# PyTorch'u CPU-only olarak kurmak için pip'i güncelle ve özel komutu çalıştır.
# Bu, en yaygın CUDA hatasını çözer.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Geri kalan kütüphaneleri requirements.txt'den kur
# Bu komut, torch'un zaten kurulu olduğunu görecek ve onu atlayacaktır.
RUN pip install --no-cache-dir -r requirements.txt

# Geri kalan tüm proje dosyalarını kopyala
COPY . .

# Uygulamayı başlatacak komutu belirt
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
