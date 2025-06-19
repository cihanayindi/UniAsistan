import os
import fitz
import numpy as np
import faiss
import json
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 1. AYARLAR VE SABİTLER ---
DATA_PATH = Path("./data")
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# HUGGINGFACE_CACHE_PATH'i kaldırdık çünkü SentenceTransformer model adını kullanarak doğru yolu kendi bulacak.
PDF_SOURCE_PATH = Path("./source_documents") # PDF'lerinizin olduğu klasör

# --- 2. YARDIMCI FONKSİYONLAR ---
def metni_temizle(text: str) -> str:
    """
    Metin içerisindeki fazla boşlukları ve satır sonlarını temizler.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_and_save_database():
    """
    Kaynak PDF'lerden metinleri çıkarır, parçalara böler, gömmeler oluşturur
    ve FAISS indeksini ile metadata'yı kaydeder.
    """
    print("Veritabanı oluşturma işlemi başlatılıyor...")

    # 1. Veri dizinini oluştur
    DATA_PATH.mkdir(exist_ok=True, parents=True)

    # 2. Embedding Modelini Yükle
    # Model zaten cache'de olduğu için, SentenceTransformer'ın model adını kullanarak
    # bunu otomatik olarak algılamasını ve yüklemesini sağlıyoruz.
    try:
        print(f"'{EMBEDDING_MODEL_NAME}' modeli yükleniyor...")
        # SentenceTransformer'ı model adıyla çağırın.
        # Bu, modelin Hugging Face cache'inde doğru konumu otomatik olarak bulmasını sağlar.
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: '{EMBEDDING_MODEL_NAME}' modeli yüklenirken bir sorun oluştu: {e}")
        print("Lütfen modelin Hugging Face cache'inde (genellikle ~/.cache/huggingface/hub/) doğru şekilde bulunduğundan emin olun.")
        return False

    # 3. PDF'lerden Metin Çıkar
    metinler_ve_kaynaklar = []
    if not PDF_SOURCE_PATH.is_dir():
        print(f"HATA: Kaynak PDF klasörü '{PDF_SOURCE_PATH}' bulunamadı!")
        print("Lütfen PDF dosyalarınızın olduğu 'source_documents' klasörünün mevcut olduğundan emin olun.")
        return False

    for dosya_yolu in PDF_SOURCE_PATH.glob("*.pdf"):
        print(f"'{dosya_yolu.name}' dosyası işleniyor...")
        try:
            doc = fitz.open(dosya_yolu)
            temiz_metin = " ".join([metni_temizle(page.get_text()) for page in doc])
            doc.close()
            if temiz_metin:
                metinler_ve_kaynaklar.append({'kaynak': dosya_yolu.name, 'icerik': temiz_metin})
        except Exception as e:
            print(f"HATA: '{dosya_yolu.name}' dosyası işlenirken bir sorun oluştu: {e}")
            continue

    if not metinler_ve_kaynaklar:
        print("UYARI: Hiçbir PDF'den metin çıkarılamadı veya 'source_documents' klasörü boş.")
        return False

    # 4. Metinleri Parçalara Böl (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    tum_parcalar = []
    for dokuman in metinler_ve_kaynaklar:
        chunks = text_splitter.split_text(dokuman['icerik'])
        for parca in chunks:
            tum_parcalar.append({'kaynak': dokuman['kaynak'], 'icerik': parca})

    if not tum_parcalar:
        print("UYARI: Hiç parça (chunk) oluşturulamadı. Bölme ayarlarınızı veya kaynak metinlerinizi kontrol edin.")
        return False

    # 5. Parçaların Gömme Vektörlerini Oluştur
    sadece_metinler = [p['icerik'] for p in tum_parcalar]
    print(f"{len(sadece_metinler)} adet metin parçası vektörlere dönüştürülüyor...")
    vektorler = model.encode(sadece_metinler, show_progress_bar=True, normalize_embeddings=True)

    # 6. FAISS İndeksi Oluştur ve Veri Ekle
    print("FAISS indeksi oluşturuluyor ve vektörler ekleniyor...")
    index = faiss.IndexFlatIP(vektorler.shape[1])
    index.add(vektorler)

    # 7. İndeksi ve Metadata'yı Kaydet
    try:
        faiss.write_index(index, str(VECTOR_DB_PATH))
        with open(METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(tum_parcalar, f, ensure_ascii=False, indent=4)
        print("Veritabanı ve metadata başarıyla kalıcı diske kaydedildi!")
        return True
    except Exception as e:
        print(f"HATA: Veritabanı veya metadata kaydedilirken bir sorun oluştu: {e}")
        return False

# --- 3. BETİĞİ ÇALIŞTIRMA ---
if __name__ == "__main__":
    if create_and_save_database():
        print("\nVeritabanı oluşturma betiği başarıyla tamamlandı.")
        print(f"FAISS indeksi kaydedildi: {VECTOR_DB_PATH}")
        print(f"Metadata kaydedildi: {METADATA_PATH}")
    else:
        print("\nVeritabanı oluşturma betiği tamamlanırken hatalar oluştu veya işlem başarısız oldu.")