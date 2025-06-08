import fitz  # PyMuPDF kütüphanesi
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. PDF'ten Metinleri Çıkarma
def extract_text_from_pdf(pdf_path):
    """Verilen yoldaki PDF dosyasının tüm metnini çıkarır."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

# 2. Metinleri Anlamlı Parçalara Bölme (Chunking)
def split_text_into_chunks(text):
    """Metni, yapay zeka için uygun boyutlarda parçalara ayırır."""
    text_splitter = RecursiveCharacterTextSplitter(
        # chunk_size: Her bir metin parçasının maksimum karakter sayısı.
        # Bu değer, kullanacağınız modelin kapasitesine göre ayarlanabilir.
        chunk_size=1000, 
        
        # chunk_overlap: Parçalar arasında kaç karakterin ortak olacağı.
        # Bu, bir cümlenin veya düşüncenin tam ortasından bölünmesini engeller.
        chunk_overlap=200,
        
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- ANA İŞLEM ---

# Okulunuzun yönerge PDF'sinin adını buraya yazın
pdf_file = "files/adustaj.pdf" 

print(f"'{pdf_file}' dosyasından metin çıkarılıyor...")
document_text = extract_text_from_pdf(pdf_file)

print("Metin parçalara ayrılıyor...")
text_chunks = split_text_into_chunks(document_text)

print(f"Toplam {len(text_chunks)} adet metin parçası (chunk) oluşturuldu.")
print("-" * 20)

# Örnek olarak ilk parçayı görelim:
# print("Oluşturulan ilk metin parçası (chunk) örneği:")
# print(text_chunks[0])

# Bu 'text_chunks' listesi, bir sonraki adımda kullanacağımız değerli verimizdir.
# Her bir elemanı, yönergenizin bir parçasıdır.

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Önceki adımdan gelen 'text_chunks' listesini burada kullandığımızı varsayalım.
# Örnek olması için kısa bir liste oluşturalım:
# text_chunks = ["...", "...", "..."] # Bu liste 2. adımdan geliyor.

# 1. Vektörleştirme Modelini Yükleme
# Türkçe ve birçok dili destekleyen, popüler ve güçlü bir model seçiyoruz.
print("Vektörleştirme modeli yükleniyor... (Bu işlem ilk seferde biraz sürebilir)")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 2. Metin Parçalarını Vektörlere Dönüştürme
print("Metin parçaları vektörlere dönüştürülüyor...")
# Bu işlem, chunk sayınıza ve bilgisayarınızın hızına göre zaman alabilir.
chunk_embeddings = model.encode(text_chunks)

print(f"Vektörlerin şekli: {chunk_embeddings.shape}")
# Çıktı şöyle görünecektir: (Toplam_chunk_sayısı, 768)
# 768, seçtiğimiz modelin her bir metni temsil etmek için kullandığı sayı miktarıdır.

# 3. Vektör Veritabanını (FAISS Index) Oluşturma ve Doldurma
dimension = chunk_embeddings.shape[1]  # Vektör boyutu (örn: 768)
index = faiss.IndexFlatL2(dimension)   # Basit bir L2 (Öklid) mesafe indeksi oluşturuyoruz.

print("Vektörler veritabanına ekleniyor...")
index.add(chunk_embeddings)

print(f"Toplam {index.ntotal} adet vektör başarıyla veritabanına eklendi.")

# --- ARAMA TESTİ ---

# Öğrencinin sorabileceği bir soru düşünelim
user_question = "bu yöneergenin amacı nedir?"

print(f"\nSoru: '{user_question}' için arama yapılıyor...")

# 1. Soruyu da aynı modelle vektöre çevir
question_embedding = model.encode([user_question])

# 2. FAISS içinde en yakın k vektörü bul (k=5 en yakın 5 sonucu getir demek)
k = 5
distances, indices = index.search(question_embedding, k)

# 3. Sonuçları göster
print("\nEn alakalı metin parçaları bulundu:")
for i in range(k):
    chunk_index = indices[0][i]
    ilgili_metin = text_chunks[chunk_index]
    
    print("-" * 20)
    print(f"{i+1}. En Yakın Sonuç (Uzaklık: {distances[0][i]:.4f})")
    print(ilgili_metin)