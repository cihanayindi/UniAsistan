# main.py dosyasının tam ve hatasız son hali

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
DATA_PATH = Path("/data")
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_PATH = DATA_PATH / "sbert_model"
PDF_SOURCE_PATH = Path("source_documents")
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'

# --- 2. FASTAPI UYGULAMASI ---
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

origins = [
    "https://uni-asistan.vercel.app",  # Vercel adresiniz
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    # Gerekirse başka adresler de eklenebilir
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # <-- Artık güncel listeyi kullanıyor
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    "embedding_model": None,
    "faiss_index": None,
    "chunks_metadata": None,
    "generative_model": None,
}

# --- 3. VERİ İŞLEME VE MODEL YÜKLEME ---
def metni_temizle(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def veritabani_ve_model_olustur():
    print("Kalıcı diskte veritabanı veya model bulunamadı. Sıfırdan oluşturuluyor...")
    DATA_PATH.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"'{EMBEDDING_MODEL_NAME}' modeli indiriliyor ve '{MODEL_PATH}' yoluna kaydediliyor...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(MODEL_PATH))
        print("Model başarıyla indirildi.")

    metinler_ve_kaynaklar = []
    if not PDF_SOURCE_PATH.is_dir():
        print(f"HATA: Kaynak PDF klasörü '{PDF_SOURCE_PATH}' bulunamadı!")
        return
    
    for dosya_yolu in PDF_SOURCE_PATH.glob("*.pdf"):
        print(f"'{dosya_yolu.name}' dosyası işleniyor...")
        doc = fitz.open(dosya_yolu)
        temiz_metin = " ".join([metni_temizle(page.get_text()) for page in doc])
        doc.close()
        if temiz_metin:
            metinler_ve_kaynaklar.append({'kaynak': dosya_yolu.name, 'icerik': temiz_metin})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    tum_parcalar = [
        {'kaynak': dokuman['kaynak'], 'icerik': parca}
        for dokuman in metinler_ve_kaynaklar
        for parca in text_splitter.split_text(dokuman['icerik'])
    ]

    if not tum_parcalar:
        print("UYARI: Hiç parça (chunk) oluşturulamadı.")
        return
        
    print(f"'{MODEL_PATH}' yolundan model yükleniyor...")
    model = SentenceTransformer(str(MODEL_PATH))
    sadece_metinler = [p['icerik'] for p in tum_parcalar]
    
    print("Metin parçaları vektörlere dönüştürülüyor...")
    vektorler = model.encode(sadece_metinler, show_progress_bar=True, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(vektorler.shape[1])
    index.add(vektorler)
    
    faiss.write_index(index, str(VECTOR_DB_PATH))
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(tum_parcalar, f, ensure_ascii=False, indent=4)
    print("Veritabanı ve model oluşturma işlemi başarıyla tamamlandı!")

@app.on_event("startup")
def startup_event():
    print("Uygulama başlatılıyor...")
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("Gemini modeli başarıyla yapılandırıldı.")

    if not all(p.exists() for p in [MODEL_PATH, VECTOR_DB_PATH, METADATA_PATH]):
        veritabani_ve_model_olustur()
    
    print("Modeller ve veritabanı kalıcı diskten yükleniyor...")
    state["embedding_model"] = SentenceTransformer(str(MODEL_PATH))
    state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        state["chunks_metadata"] = json.load(f)
    print("Tüm modeller ve veritabanı başarıyla yüklendi.")

# --- 4. API ENDPOINT'LERİ ---
@app.get("/")
def read_root():
    return {"status": "UniAsistan API çalışıyor!"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not all(state.values()):
        raise HTTPException(status_code=503, detail="Modeller veya veritabanı yüklenemedi. Lütfen logları kontrol edin.")
    
    question_vector = state["embedding_model"].encode([request.question], normalize_embeddings=True)
    distances, indices = state["faiss_index"].search(question_vector, 5)
    
    context_parts = [state["chunks_metadata"][i]['icerik'] for i in indices[0]]
    sources = {state["chunks_metadata"][i]['kaynak'] for i in indices[0]}
    context = "\n---\n".join(context_parts)

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
    
    try:
        response = state["generative_model"].generate_content(prompt)
        answer = response.text
    except Exception as e:
        print(f"Gemini API hatası: {e}")
        raise HTTPException(
            status_code=500,
            detail="Cevap üretilirken bir hata oluştu."
        )

    return AnswerResponse(answer=answer, sources=list(sources))