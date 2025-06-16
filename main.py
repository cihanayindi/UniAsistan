# Libraries
import os
import faiss
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# --- Constants and Settings ---
# Bu yollar, Docker imajı içindeki uygulama kök dizinine göre belirlenmiştir.
DATA_PATH = Path("./data") 
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
# Bu model artık lokalden değil, doğrudan kütüphane üzerinden yüklenecek.
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

# --- FastAPI App ---
app = FastAPI(
    title="UniAsistan API",
    description="Adnan Menderes Üniversitesi için RAG tabanlı yardım asistanı.",
    version="1.0.0"
)

# CORS Middleware
origins = [
    "https://uni-asistan.vercel.app",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "null",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state to hold models and data
state = {
    "embedding_model": None,
    "faiss_index": None,
    "chunks_metadata": None,
    "generative_model": None,
}

@app.on_event("startup")
def startup_event():
    """
    Uygulama başlarken modelleri ve önceden oluşturulmuş veritabanını yükler.
    Artık veritabanı OLUŞTURMAZ, sadece YÜKLER.
    """
    print("Uygulama başlatılıyor...")
    load_dotenv()
    
    # 1. Gemini Modelini Yapılandır
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("UYARI: GEMINI_API_KEY ortam değişkeni bulunamadı. API çalışmayabilir.")
    else:
        genai.configure(api_key=api_key)
        state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("Gemini modeli başarıyla yapılandırıldı.")

    try:
        # 2. Embedding Modelini Yükle
        print(f"Embedding modeli '{EMBEDDING_MODEL_NAME}' yükleniyor...")
        # Model, kütüphane tarafından yönetilen bir cache dizinine indirilecek.
        state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 3. FAISS Veritabanını Yükle
        print(f"FAISS index'i '{VECTOR_DB_PATH}' yolundan yükleniyor...")
        state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))
        
        # 4. Metadata'yı Yükle
        print(f"Metadata '{METADATA_PATH}' yolundan yükleniyor...")
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            state["chunks_metadata"] = json.load(f)
            
        print("Uygulama başarıyla başlatıldı ve kullanıma hazır!")

    except FileNotFoundError as e:
        print(f"KRİTİK HATA: Gerekli bir dosya bulunamadı: {e}")
        print("Lütfen 'create_database_local.py' scriptini çalıştırdığınızdan ve 'data' klasörünün Docker imajına eklendiğinden emin olun.")
        # Hata durumunda state'in bazı elemanları None kalacak ve endpoint'ler hata verecektir.
    except Exception as e:
        print(f"Başlatma sırasında beklenmedik bir hata oluştu: {e}")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "UniAsistan API çalışıyor!"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not all([
        state["embedding_model"],
        state["faiss_index"],
        state["chunks_metadata"],
        state["generative_model"]
    ]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uygulama başlatılırken bir sorun oluştu veya henüz hazır değil. Lütfen logları kontrol edin."
        )

    user_question_lower = request.question.lower().strip()
    meta_questions_keywords = {
        ("sen kimsin", "kimsin sen", "nesin sen", "sen nesin"): "Ben Cihan Ayindi tarafından ADÜ Öğrencilerine Öğrenci işlerinde sorulabilecek konularda öğrencilere yardımcı olması amacıyla geliştirilmiş bir RAG sistemiyim.",
        ("kim geliştirdi", "geliştiricin kim", "seni kim yaptı"): "Ben Cihan Ayindi tarafından geliştirildim.",
        ("amacın ne", "ne işe yararsın", "görevin ne"): "Amacım, Adnan Menderes Üniversitesi öğrencilerine öğrenci işleriyle ilgili konularda yardımcı olmaktır."
    }
    for keywords, answer in meta_questions_keywords.items():
        if any(keyword in user_question_lower for keyword in keywords):
            return AnswerResponse(answer=answer, sources=[])

    # RAG süreci
    question_vector = state["embedding_model"].encode(
        [request.question], normalize_embeddings=True
    ).astype(np.float32)
    
    distances, indices = state["faiss_index"].search(question_vector, 7)
    
    if not indices.size:
        return AnswerResponse(answer="Sorunuza uygun bir içerik bulamadım.", sources=[])

    context_parts = [state["chunks_metadata"][i]['icerik'] for i in indices[0]]
    sources = {state["chunks_metadata"][i]['kaynak'] for i in indices[0]}
    context = "\n---\n".join(context_parts)

    prompt = f"""Sen Adnan Menderes Üniversitesi Öğrenci İşleri için bir yardım asistanısın. Görevin, sadece ve sadece aşağıda sana verdiğim 'Bağlam' metinlerini kullanarak kullanıcının sorusuna cevap vermektir. Eğer cevap bağlamda açıkça yoksa veya emin değilsen, 'Bu konuda bilgi sahibi değilim.' de. Cevabın kesinlikle bu bağlamın dışına çıkmamalıdır. Cevaplarını nazik, anlaşılır ve kısa tut.
---
Bağlam:
{context}
---
Kullanıcı Sorusu: {request.question}
Cevap:"""
    
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
