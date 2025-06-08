# --- Gerekli Kütüphanelerin Import Edilmesi ---
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import json
import re
from pathlib import Path

# Sunucu ve API için gerekli kütüphaneler
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Dil modelleri ve çevre değişkenleri için kütüphaneler
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. AYARLAR VE SABİTLER ---

# Kalıcı diskimizin bağlandığı yolu belirliyoruz.
# Fly.io'da bu klasör, her yeniden başlatmada silinmez.
DATA_PATH = Path("/data")
PDF_SOURCE_PATH = Path("source_documents") # PDF'leri ilk başta buradan alacağız
DOCUMENTS_PATH = DATA_PATH / "documents"    # PDF'leri kalıcı olarak buraya taşıyacağız
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"

# Kullanacağımız modellerin adları
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'

# --- 2. FASTAPI UYGULAMASI VE VERİ MODELLERİ ---

# Pydantic modelleri, API'ye gelen ve giden verinin yapısını tanımlar.
# Bu, hataları önler ve kodu daha okunabilir kılar.
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

# FastAPI uygulamasını oluşturuyoruz.
app = FastAPI(
    title="UniAsistan API",
    description="Adnan Menderes Üniversitesi için RAG tabanlı yardım asistanı.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) Ayarları
# Front-end'in (Vercel) bu back-end'e (Fly.io) erişmesine izin verir.
origins = [
    "*"  # Test için tüm kaynaklara izin ver, canlıda Vercel adresini yazmak daha güvenli
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uygulama genelinde kullanılacak modelleri ve veriyi tutacak olan global state.
# Bu sayede modeller her istekte değil, sadece bir kez yüklenir.
state = {
    "embedding_model": None,
    "faiss_index": None,
    "chunks_metadata": None,
    "generative_model": None,
}

# --- 3. VERİ İŞLEME VE MODEL YÜKLEME ---

def metni_temizle(text: str) -> str:
    """PDF'ten gelen kirli metni temizler."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def veritabani_olustur():
    """
    /data/documents klasöründeki PDF'leri işler, vektör veritabanını oluşturur
    ve kalıcı diske kaydeder. Bu işlem sadece bir kez yapılır.
    """
    print("Vektör veritabanı bulunamadı. Sıfırdan oluşturuluyor...")
    
    # 1. Kaynak PDF'leri /data klasörüne taşı
    DOCUMENTS_PATH.mkdir(exist_ok=True)
    if PDF_SOURCE_PATH.exists():
        for pdf_file in PDF_SOURCE_PATH.glob("*.pdf"):
            print(f"{pdf_file.name} dosyası /data/documents içine kopyalanıyor...")
            (DOCUMENTS_PATH / pdf_file.name).write_bytes(pdf_file.read_bytes())
    
    # 2. PDF'leri oku ve metinleri çıkar
    metinler_ve_kaynaklar = []
    pdf_files = list(DOCUMENTS_PATH.glob("*.pdf"))
    if not pdf_files:
        print("UYARI: İşlenecek PDF dosyası bulunamadı. Veritabanı boş olacak.")
        return

    for dosya_yolu in pdf_files:
        print(f"'{dosya_yolu.name}' dosyası işleniyor...")
        doc = fitz.open(dosya_yolu)
        temiz_metin = " ".join([metni_temizle(page.get_text()) for page in doc])
        doc.close()
        if temiz_metin:
            metinler_ve_kaynaklar.append({'kaynak': dosya_yolu.name, 'icerik': temiz_metin})

    # 3. Metinleri parçalara ayır (chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    tum_parcalar = []
    for dokuman in metinler_ve_kaynaklar:
        parcalar = text_splitter.split_text(dokuman['icerik'])
        for parca in parcalar:
            tum_parcalar.append({'kaynak': dokuman['kaynak'], 'icerik': parca})

    if not tum_parcalar:
        print("UYARI: Metinlerden hiç parça (chunk) oluşturulamadı. Veritabanı boş olacak.")
        return
        
    # 4. Parçaları vektörlere dönüştür ve FAISS veritabanını oluştur
    print(f"'{EMBEDDING_MODEL_NAME}' modeli yükleniyor...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    sadece_metinler = [p['icerik'] for p in tum_parcalar]
    
    print("Metin parçaları vektörlere dönüştürülüyor...")
    vektorler = model.encode(sadece_metinler, show_progress_bar=True, normalize_embeddings=True)
    
    dimension = vektorler.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner Product için IndexFlatIP daha iyi olabilir
    index.add(vektorler)
    
    # 5. Veritabanını ve metadata'yı kalıcı diske kaydet
    print(f"İndeks '{VECTOR_DB_PATH}' dosyasına kaydediliyor...")
    faiss.write_index(index, str(VECTOR_DB_PATH))
    
    print(f"Metadata '{METADATA_PATH}' dosyasına kaydediliyor...")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(tum_parcalar, f, ensure_ascii=False, indent=4)
        
    print("Veritabanı oluşturma işlemi başarıyla tamamlandı!")


@app.on_event("startup")
def startup_event():
    """
    Uygulama ilk başladığında SADECE BİR KEZ çalışacak olan fonksiyon.
    Modelleri ve veritabanını hafızaya yükler.
    """
    print("Uygulama başlatılıyor... Modeller ve veritabanı yükleniyor.")
    
    # 1. Gemini API Anahtarını Yükle
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("UYARI: GEMINI_API_KEY bulunamadı. Üretken model çalışmayacak.")
    else:
        genai.configure(api_key=api_key)
        state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("Gemini modeli başarıyla yapılandırıldı.")

    # 2. Vektör veritabanı kontrolü ve yüklenmesi
    if not VECTOR_DB_PATH.exists() or not METADATA_PATH.exists():
        veritabani_olustur()

    # 3. Gerekli dosyaları ve modelleri hafızaya (state'e) yükle
    if VECTOR_DB_PATH.exists() and METADATA_PATH.exists():
        print("FAISS veritabanı ve embedding modeli yükleniyor...")
        state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            state["chunks_metadata"] = json.load(f)
        state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Tüm modeller ve veritabanı başarıyla yüklendi.")
    else:
        print("UYARI: Veritabanı dosyaları bulunamadığı için arama fonksiyonu çalışmayacak.")
        
# --- 4. API ENDPOINT'LERİ ---

@app.get("/", status_code=status.HTTP_200_OK, tags=["Sağlık Kontrolü"])
def read_root():
    """API'nin çalışıp çalışmadığını kontrol etmek için basit bir endpoint."""
    return {"status": "UniAsistan API çalışıyor!"}

@app.post("/ask", response_model=AnswerResponse, tags=["Soru-Cevap"])
def ask_question(request: QuestionRequest):
    """
    Front-end'den gelen soruyu alır, RAG işlemini gerçekleştirir ve cevabı döndürür.
    """
    # Gerekli modellerin yüklenip yüklenmediğini kontrol et
    if not all([state["faiss_index"], state["embedding_model"], state["generative_model"]]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modeller veya veritabanı yüklenemedi. Lütfen logları kontrol edin."
        )

    # 1. Gelen soruyu vektöre çevir
    question_vector = state["embedding_model"].encode([request.question], normalize_embeddings=True)
    
    # 2. FAISS'te en yakın K parçayı bul (örneğin 5 tane)
    k = 5
    distances, indices = state["faiss_index"].search(question_vector, k)
    
    # 3. Bulunan parçaların metinlerini ve kaynaklarını al
    context_parts = []
    sources = set()
    for i in range(k):
        chunk_index = indices[0][i]
        chunk = state["chunks_metadata"][chunk_index]
        context_parts.append(chunk['icerik'])
        sources.add(chunk['kaynak'])
        
    context = "\n---\n".join(context_parts)

    # 4. Gemini için Prompt oluştur
    prompt = f"""
    Sen Adnan Menderes Üniversitesi Öğrenci İşleri için bir yardım asistanısın.
    Görevin, sadece ve sadece aşağıda sana verdiğim 'Bağlam' metinlerini kullanarak kullanıcının sorusuna cevap vermektir.
    Cevabın kesinlikle bu bağlamın dışına çıkmamalıdır. Eğer cevap bağlamda yoksa, 'Bu konuda bilgi sahibi değilim.' de.
    Cevaplarını nazik, anlaşılır ve kısa tut.
    
    ---
    Bağlam:
    {context}
    ---
    
    Kullanıcı Sorusu: {request.question}
    
    Cevap:
    """

    # 5. Gemini API'yi çağır ve cevabı üret
    try:
        generative_model = state["generative_model"]
        response = generative_model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Gemini API hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cevap üretilirken bir hata oluştu."
        )

    return AnswerResponse(answer=answer, sources=list(sources))

