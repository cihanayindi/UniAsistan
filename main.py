# --- Gerekli Kütüphanelerin Import Edilmesi ---
import os
import fitz
import numpy as np
import faiss
import json
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. AYARLAR VE SABİTLER ---

# Kalıcı disk (/data) üzerindeki yolları tanımlıyoruz
DATA_PATH = Path("/data")
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_PATH = DATA_PATH / "sbert_model"  # Modelin kalıcı diskteki yeri

# Orjinal PDF'lerin Docker imajı içindeki yeri
PDF_SOURCE_PATH = Path("source_documents")

GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'


# --- 2. FASTAPI UYGULAMASI VE VERİ MODELLERİ ---
# (Bu kısım aynı kalıyor)
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

app = FastAPI(
    title="UniAsistan API",
    description="Adnan Menderes Üniversitesi için RAG tabanlı yardım asistanı.",
    version="1.0.0"
)
origins = ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

state = { "embedding_model": None, "faiss_index": None, "chunks_metadata": None, "generative_model": None }


# --- 3. VERİ İŞLEME VE MODEL YÜKLEME ---

def metni_temizle(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Bu fonksiyon artık sadece gerektiğinde, uygulama başlangıcında çağrılacak.
def veritabani_ve_model_olustur():
    print("Kalıcı diskte veritabanı veya model bulunamadı. Sıfırdan oluşturuluyor...")
    
    # Modelin indirilmesi
    if not MODEL_PATH.exists():
        print(f"'{EMBEDDING_MODEL_NAME}' modeli indiriliyor ve '{MODEL_PATH}' yoluna kaydediliyor...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(MODEL_PATH)
        print("Model başarıyla indirildi.")
    
    # Veritabanının oluşturulması
    DATA_PATH.mkdir(exist_ok=True)
    metinler_ve_kaynaklar = []
    pdf_files = list(PDF_SOURCE_PATH.glob("*.pdf"))
    if not pdf_files:
        print("UYARI: İşlenecek PDF dosyası bulunamadı.")
        return
    for dosya_yolu in pdf_files:
        print(f"'{dosya_yolu.name}' dosyası işleniyor...")
        doc = fitz.open(dosya_yolu)
        temiz_metin = " ".join([metni_temizle(page.get_text()) for page in doc])
        doc.close()
        if temiz_metin:
            metinler_ve_kaynaklar.append({'kaynak': dosya_yolu.name, 'icerik': temiz_metin})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    tum_parcalar = []
    for dokuman in metinler_ve_kaynaklar:
        parcalar = text_splitter.split_text(dokuman['icerik'])
        for parca in parcalar:
            tum_parcalar.append({'kaynak': dokuman['kaynak'], 'icerik': parca})

    if not tum_parcalar:
        print("UYARI: Hiç parça (chunk) oluşturulamadı.")
        return
        
    print(f"'{MODEL_PATH}' yolundan model yükleniyor...")
    model = SentenceTransformer(MODEL_PATH)
    sadece_metinler = [p['icerik'] for p in tum_parcalar]
    
    print("Metin parçaları vektörlere dönüştürülüyor...")
    vektorler = model.encode(sadece_metinler, show_progress_bar=True, normalize_embeddings=True)
    
    dimension = vektorler.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vektorler)
    
    print(f"İndeks '{VECTOR_DB_PATH}' dosyasına kaydediliyor...")
    faiss.write_index(index, str(VECTOR_DB_PATH))
    
    print(f"Metadata '{METADATA_PATH}' dosyasına kaydediliyor...")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(tum_parcalar, f, ensure_ascii=False, indent=4)
    print("Veritabanı oluşturma işlemi başarıyla tamamlandı!")

@app.on_event("startup")
def startup_event():
    print("Uygulama başlatılıyor...")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("Gemini modeli başarıyla yapılandırıldı.")

    # Kalıcı diskte model ve veritabanı var mı diye kontrol et
    if not MODEL_PATH.exists() or not VECTOR_DB_PATH.exists() or not METADATA_PATH.exists():
        # Eğer biri bile eksikse, hepsini sıfırdan oluştur
        veritabani_ve_model_olustur()
    
    print("Modeller ve veritabanı kalıcı diskten yükleniyor...")
    state["embedding_model"] = SentenceTransformer(MODEL_PATH)
    state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        state["chunks_metadata"] = json.load(f)
    print("Tüm modeller ve veritabanı başarıyla yüklendi.")

# --- 4. API ENDPOINT'LERİ ---
# (Bu kısım aynı kalıyor)
@app.get("/", ...)
@app.post("/ask", ...)