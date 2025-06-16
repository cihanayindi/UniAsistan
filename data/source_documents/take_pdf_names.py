import os

# 'files' klasörünün yolu
current_directory = os.path.join(os.getcwd(), "files")

# Dosyaları al
files = os.listdir(current_directory)

# PDF dosyalarını filtrele ve yazdır
for file in files:
    if file.lower().endswith(".pdf"):
        print(file)
