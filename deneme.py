import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import re  ### YENİ EKLENDİ ### -> Metin temizleme için 'regular expression' kütüphanesi

# --- 1. AYARLAR VE SABİTLER ---
PDF_KLASORU = "source_documents"
VEKTOR_DB_YOLU = "vektor/faiss_index.bin"
METADATA_YOLU = "vektor/chunks_metadata.json"
MODEL_ADI = 'paraphrase-multilingual-mpnet-base-v2'

# --- 2. VERİ İŞLEME FONKSİYONLARI ---

### YENİ EKLENDİ ###
def metni_temizle(text):
    """
    PDF'ten gelen kirli metni temizler.
    - Çoklu boşlukları ve satır atlamaları tek boşluğa indirir.
    - Başta ve sonda kalan boşlukları kaldırır.
    """
    # Birden fazla boşluk veya satır atlama karakterini (newline, tab vb.) tek boşluğa çevirir.
    text = re.sub(r'\s+', ' ', text)
    # Metnin başındaki ve sonundaki olası boşlukları temizler.
    text = text.strip()
    return text

### GÜNCELLENDİ ###
def pdf_metinlerini_al(klasor_yolu):
    """
    Belirtilen klasördeki tüm PDF dosyalarını okur, metinlerini temizler
    ve kaynak dosya adıyla birlikte bir liste olarak döndürür.
    """
    metinler_ve_kaynaklar = []
    for dosya_adi in os.listdir(klasor_yolu):
        if dosya_adi.lower().endswith('.pdf'):
            print(f"'{dosya_adi}' dosyası işleniyor...")
            dosya_yolu = os.path.join(klasor_yolu, dosya_adi)
            doc = fitz.open(dosya_yolu)
            ham_metin = ""
            for page in doc:
                ham_metin += page.get_text()
            doc.close()
            
            # Metni aldıktan sonra temizleme fonksiyonunu çağırıyoruz.
            temiz_metin = metni_temizle(ham_metin)
            
            metinler_ve_kaynaklar.append({'kaynak': dosya_adi, 'icerik': temiz_metin})
    return metinler_ve_kaynaklar

def metni_parcalara_ayir(metinler_ve_kaynaklar):
    """
    Ham metinleri, kaynak bilgisiyle birlikte daha küçük ve yönetilebilir
    parçalara (chunk) ayırır.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    tum_parcalar = []
    for dokuman in metinler_ve_kaynaklar:
        parcalar = text_splitter.split_text(dokuman['icerik'])
        for parca in parcalar:
            tum_parcalar.append({'kaynak': dokuman['kaynak'], 'icerik': parca})
    return tum_parcalar

def veritabani_olustur_ve_kaydet(parcalar):
    """
    Metin parçalarını vektörlere dönüştürür, bir FAISS veritabanı oluşturur
    ve hem indeksi hem de metin verisini diske kaydeder.
    """
    print("Vektörleştirme modeli yükleniyor...")
    model = SentenceTransformer(MODEL_ADI)
    
    print("Metin parçaları vektörlere dönüştürülüyor...")
    sadece_metinler = [p['icerik'] for p in parcalar]
    vektorler = model.encode(sadece_metinler, show_progress_bar=True)

    print("FAISS veritabanı oluşturuluyor...")
    dimension = vektorler.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vektorler)

    print(f"'{VEKTOR_DB_YOLU}' dosyasına indeks kaydediliyor...")
    faiss.write_index(index, VEKTOR_DB_YOLU)

    print(f"'{METADATA_YOLU}' dosyasına metin verisi kaydediliyor...")
    with open(METADATA_YOLU, 'w', encoding='utf-8') as f:
        json.dump(parcalar, f, ensure_ascii=False, indent=4)
        
    print("\nVeritabanı oluşturma ve kaydetme işlemi başarıyla tamamlandı!")

# --- 3. ARAMA FONKSİYONLARI ---

def veritabani_ve_model_yukle():
    """
    Kaydedilmiş FAISS indeksini, metin verisini ve modeli yükler.
    Bu fonksiyon, asıl uygulamanızda kullanılacak.
    """
    print("Kaydedilmiş veritabanı ve model yükleniyor...")
    index = faiss.read_index(VEKTOR_DB_YOLU)
    with open(METADATA_YOLU, 'r', encoding='utf-8') as f:
        parcalar = json.load(f)
    model = SentenceTransformer(MODEL_ADI)
    return index, parcalar, model

def arama_yap(soru, index, parcalar, model, en_yakin_k=5):
    """
    Verilen soruya göre veritabanında anlamsal arama yapar.
    """
    soru_vektoru = model.encode([soru])
    mesafeler, indisler = index.search(soru_vektoru, en_yakin_k)
    
    print(f"\nSoru: '{soru}' için arama yapılıyor...")
    print("\nEn alakalı metin parçaları bulundu:")
    
    sonuclar = []
    for i in range(en_yakin_k):
        parca_indisi = indisler[0][i]
        ilgili_parca = parcalar[parca_indisi]
        sonuclar.append(ilgili_parca)
        
        print("-" * 20)
        print(f"{i+1}. En Yakın Sonuç (Kaynak: {ilgili_parca['kaynak']}, Uzaklık: {mesafeler[0][i]:.4f})")
        print(ilgili_parca['icerik'])
        
    return sonuclar

# --- ANA ÇALIŞTIRMA BLOKLARI ---
if __name__ == '__main__':
    # Bu blok, veritabanını SIFIRDAN OLUŞTURMAK için kullanılır.
    # Metin temizleme özelliğini eklediğiniz için, daha kaliteli bir veritabanı
    # oluşturmak adına bu bölümü tekrar çalıştırmalısınız.
    
    print("--- Veritabanı Oluşturma ve Temizleme Modu ---")
    ham_metinler = pdf_metinlerini_al(PDF_KLASORU)
    metin_parcalari = metni_parcalara_ayir(ham_metinler)
    veritabani_olustur_ve_kaydet(metin_parcalari)
    
    print("\n" + "="*50 + "\n")

    # Bu blok, oluşturulmuş yeni ve temiz veritabanı üzerinde ARAMA TESTİ yapmak için kullanılır.
    # Asıl uygulamanız bu mantıkla çalışacak.
    
    print("--- Arama Test Modu ---")
    yuklenmis_index, yuklenmis_parcalar, yuklenmis_model = veritabani_ve_model_yukle()
    arama_yap("Ön lisans ve lisans yönetmeliğinin amacı nedir?", yuklenmis_index, yuklenmis_parcalar, yuklenmis_model)