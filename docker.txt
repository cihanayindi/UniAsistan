# Python'un resmi, hafif bir sürümünü temel alarak başlıyoruz.
FROM python:3.11-slim

# Çalışma ortamı olarak /code klasörünü ayarlıyoruz.
WORKDIR /code

#requirements.txt dosyasını kopyalayıp bağımlılıkları yüklüyoruz.
# Bu adımı ayrı yapmak, Docker'ın katman önbellekleme özelliğinden yararlanarak
# sonraki build'leri hızlandırır.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Projenin geri kalan tüm dosyalarını /code klasörüne kopyalıyoruz.
COPY . .

# Fly.io'nun uygulamaya erişeceği portu belirtiyoruz.
EXPOSE 8080

# Uygulamayı çalıştıracak komut.
# main:app -> main.py dosyasındaki "app" isimli FastAPI uygulamasını çalıştır.
# --host 0.0.0.0 -> Uygulamanın dışarıdan gelen isteklere açık olmasını sağlar.
# --port 8080 -> Fly.io'nun beklediği port.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]