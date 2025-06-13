#!/bin/sh

# Bu script, sunucu başladığında çalışacak ve bize teşhis bilgisi verecek.

echo "--- Başlatıcı Script Çalışıyor ---"
echo "Kalıcı disk (/data) içeriği kontrol ediliyor:"
ls -R /data

echo "--- Kontrol Tamamlandı. Python uygulaması başlatılıyor... ---"

# Şimdi asıl uygulamayı başlat
uvicorn main:app --host 0.0.0.0 --port 8080