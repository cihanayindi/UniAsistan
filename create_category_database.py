import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --- 1. AYARLAR VE SABİTLER ---
# Ana veri klasörü, orijinal betikle aynı kalabilir.
DATA_PATH = Path("./dataV5") 
# Kategori veritabanı dosyaları için çakışmayı önlemek adına farklı isimler kullanıyoruz.
VECTOR_DB_PATH = DATA_PATH / "category_faiss_index.bin"
METADATA_PATH = DATA_PATH / "category_metadata.json"
# Kaynak JSON dosyası
KATEGORI_ACIKLAMALARI_PATH = Path("./source_documents/categories.json")
# Model adı, orijinal betikle tutarlı olmalı.
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

def create_category_database():
    """
    Kategori açıklamalarını JSON'dan okur, gömmeler oluşturur
    ve kategori için FAISS indeksini ile metadata'yı kaydeder.
    """
    print("Kategori veritabanı oluşturma işlemi başlatılıyor...")

    # 1. Veri dizinini oluştur (zaten varsa bir şey yapmaz)
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    # 2. Kaynak JSON dosyasını kontrol et
    if not KATEGORI_ACIKLAMALARI_PATH.is_file():
        print(f"HATA: Kategori açıklamalarını içeren '{KATEGORI_ACIKLAMALARI_PATH}' dosyası bulunamadı!")
        return False
        
    # 3. Embedding Modelini Yükle
    # Model zaten cache'de olduğu için tekrar indirilmeyecek.
    try:
        print(f"'{EMBEDDING_MODEL_NAME}' modeli yükleniyor...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: '{EMBEDDING_MODEL_NAME}' modeli yüklenirken bir sorun oluştu: {e}")
        return False

    # 4. Kategori Açıklamalarını JSON'dan Oku
    with open(KATEGORI_ACIKLAMALARI_PATH, 'r', encoding='utf-8') as f:
        kategoriler = json.load(f)

    # Metadata (sadece kategori adları) ve gömülecek metinleri (açıklamalar) ayır
    kategori_adlari = list(kategoriler.keys())
    kategori_aciklamalari = list(kategoriler.values())

    if not kategori_aciklamalari:
        print("UYARI: JSON dosyasından hiç kategori açıklaması okunamadı.")
        return False

    # 5. Açıklamaların Gömme Vektörlerini Oluştur
    print(f"{len(kategori_aciklamalari)} adet kategori açıklaması vektörlere dönüştürülüyor...")
    vektorler = model.encode(kategori_aciklamalari, show_progress_bar=True, normalize_embeddings=True)

    # 6. FAISS İndeksi Oluştur ve Veri Ekle
    print("FAISS indeksi oluşturuluyor ve kategori vektörleri ekleniyor...")
    index = faiss.IndexFlatIP(vektorler.shape[1])
    index.add(vektorler.astype(np.float32)) # FAISS için float32'ye çevirmek en iyisidir.

    # 7. İndeksi ve Metadata'yı Kaydet
    try:
        faiss.write_index(index, str(VECTOR_DB_PATH))
        # Metadata olarak sadece kategori adlarının listesini kaydediyoruz.
        # İndeksteki N. vektör, bu listedeki N. kategoriye aittir.
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(kategori_adlari, f, ensure_ascii=False, indent=4)
        print("Kategori veritabanı ve metadata başarıyla kalıcı diske kaydedildi!")
        return True
    except Exception as e:
        print(f"HATA: Veritabanı veya metadata kaydedilirken bir sorun oluştu: {e}")
        return False

# --- 3. BETİĞİ ÇALIŞTIRMA ---
if __name__ == "__main__":
    if create_category_database():
        print("\nKategori veritabanı oluşturma betiği başarıyla tamamlandı.")
        print(f"Kategori FAISS indeksi kaydedildi: {VECTOR_DB_PATH}")
        print(f"Kategori metadata kaydedildi: {METADATA_PATH}")
    else:
        print("\nKategori veritabanı oluşturma betiği tamamlanırken hatalar oluştu veya işlem başarısız oldu.")