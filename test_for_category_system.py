import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 1. AYARLAR VE SABİTLER ---
# Bu yolların, oluşturucu betikteki yollarla aynı olduğundan emin olun.
DATA_PATH = Path("./data")
VECTOR_DB_PATH = DATA_PATH / "category_faiss_index.bin"
METADATA_PATH = DATA_PATH / "category_metadata.json"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

def siniflandirmayi_test_et():
    """
    Kategori sınıflandırma sistemini yükler ve önceden tanımlanmış
    test sorgularıyla performansını test eder.
    """
    print("Kategori sınıflandırma test betiği başlatılıyor...")

    # 1. Gerekli dosyaların varlığını kontrol et
    if not VECTOR_DB_PATH.is_file() or not METADATA_PATH.is_file():
        print(f"HATA: Veritabanı dosyaları bulunamadı!")
        print(f"Lütfen '{VECTOR_DB_PATH}' ve '{METADATA_PATH}' dosyalarının mevcut olduğundan emin olun.")
        print("Eğer dosyalar yoksa, önce 'kategori_veritabani_olustur.py' betiğini çalıştırın.")
        return

    # 2. Modeli, FAISS indeksini ve metadata'yı yükle
    try:
        print(f"'{EMBEDDING_MODEL_NAME}' modeli yükleniyor...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print("FAISS indeksi ve metadata yükleniyor...")
        index = faiss.read_index(str(VECTOR_DB_PATH))
        
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            kategori_adlari = json.load(f)
            
        print("Tüm bileşenler başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Gerekli bileşenler yüklenirken bir sorun oluştu: {e}")
        return

    # 3. Test edilecek sorguları tanımla
    test_sorgulari = [
        "Mezuniyet için gerekenler nelerdir?",
    ]
    
    print("\n--- TEST SONUÇLARI ---")
    
    # 4. Her bir sorguyu sınıflandır ve sonucu yazdır
    for sorgu in test_sorgulari:
        # Sorguyu vektöre dönüştür
        sorgu_vektoru = model.encode(sorgu, normalize_embeddings=True)
        
        # FAISS'te arama yapmak için vektörü uygun formata getir (2D ve float32)
        sorgu_vektoru_np = np.array([sorgu_vektoru]).astype('float32')
        
        # En yakın 3 kategoriyi bul (k=3)
        # distances: Benzerlik skorları (IP: Inner Product olduğu için büyük olan daha iyi)
        # indices: Bulunan vektörlerin indeksleri
        distances, indices = index.search(sorgu_vektoru_np, k=3)
        
        print(f"\n-> Sorgu: '{sorgu}'")
        print("   Tahmin Edilen Kategoriler:")
        
        # Bulunan sonuçları ekrana yazdır
        for i in range(len(indices[0])):
            bulunan_indeks = indices[0][i]
            bulunan_kategori = kategori_adlari[bulunan_indeks]
            benzerlik_skoru = distances[0][i]
            
            # İlk sonucu en olası tahmin olarak işaretle
            isaret = "(En Olası Tahmin)" if i == 0 else ""
            print(f"     {i+1}. {bulunan_kategori} (Skor: {benzerlik_skoru:.4f}) {isaret}")
            
# --- Betiği Çalıştır ---
if __name__ == "__main__":
    siniflandirmayi_test_et()