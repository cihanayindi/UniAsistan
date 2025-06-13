FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . .

# Yeni eklenen satır: script'i çalıştırılabilir yap
RUN chmod +x /code/start.sh

EXPOSE 8080

# CMD komutunu yeni script'i çalıştıracak şekilde değiştir
CMD ["/code/start.sh"]