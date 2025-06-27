import os
import requests

# PDF'lerin kaydedileceği klasör
download_folder = r"\UniAsistan\source_documents\yeni_pdf"

# Link dosyasının adı
links_file = "link.txt"

# Klasör yoksa oluştur
os.makedirs(download_folder, exist_ok=True)

# link dosyasını oku
with open(links_file, "r", encoding="utf-8") as file:
    links = file.readlines()

# Her bir PDF linkini indir
for i, link in enumerate(links):
    link = link.strip()
    if not link:
        continue
    try:
        response = requests.get(link)
        response.raise_for_status()

        # Dosya adı linkin son kısmından alınır
        filename = link.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename = f"file_{i+1}.pdf"

        file_path = os.path.join(download_folder, filename)

        # Dosyayı yaz
        with open(file_path, "wb") as f:
            f.write(response.content)

        print(f"[✓] İndirildi: {filename}")
    except Exception as e:
        print(f"[!] Hata oluştu ({link}): {e}")
