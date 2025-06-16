# Libraries
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

# Constants and Settings
if os.getenv("ENV") == "prod":
    DATA_PATH = Path("/data") # Production path for Fly.io
else:
    DATA_PATH = Path("./data")  # Local development path

VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
MODEL_PATH = Path("/app/models/sbert_model")
PDF_SOURCE_PATH = DATA_PATH / "source_documents"
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'  # Google Gemini model name
MIN_CHUNK_LENGTH = 30 # Minimum length of text chunks to consider

# FastAPI application instance
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
    "https://uni-asistan.vercel.app",  # Our Vercel address
    "http://localhost:8080",
    "http://127.0.1:8080",
    "null",  # Allow null origin for local testing
    # Additional origins can be added if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Using the updated list
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

# Data preparation and model loading would go here
def clear_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_database_and_models():
    print("Kalıcı diskte veritabanı veya model bulunamadı. Sıfırdan oluşturuluyor...")
    DATA_PATH.mkdir(exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"'{EMBEDDING_MODEL_NAME}' modeli indiriliyor ve '{MODEL_PATH}' yoluna kaydediliyor...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        model.save(str(MODEL_PATH))
        print(f"'{EMBEDDING_MODEL_NAME}' modeli başarıyla kaydedildi.")

    metinler_ve_kaynaklar = []
    if not PDF_SOURCE_PATH.is_dir():
        print(f"'{PDF_SOURCE_PATH}' klasörü bulunamadı.")
        return
    
    for dosya_yolu in PDF_SOURCE_PATH.glob("*.pdf"):
        print(f"'{dosya_yolu.name}' dosyası işleniyor...")
        doc = fitz.open(dosya_yolu)
        temiz_metin = " ".join([clear_text(page.get_text()) for page in doc])
        doc.close()
        if temiz_metin:
            metinler_ve_kaynaklar.append({'kaynak': dosya_yolu.name, 'icerik': temiz_metin})
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,      
        chunk_overlap=100,   
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    unfiltered_all_chunks_data = []
    for doc in metinler_ve_kaynaklar:
        # Önce tüm chunkları alalım
        raw_chunks = text_splitter.split_text(doc['icerik'])
        for chunk_text in raw_chunks:
            unfiltered_all_chunks_data.append({'kaynak': doc['kaynak'], 'icerik': chunk_text})

    all_chunks = []
    for chunk_data in unfiltered_all_chunks_data:
        if len(chunk_data['icerik'].strip()) >= MIN_CHUNK_LENGTH:
            all_chunks.append(chunk_data)

    print(f"Toplam chunk sayısı (filtrelenmeden önce): {len(unfiltered_all_chunks_data)}")
    print(f"Toplam chunk sayısı (filtrelendikten sonra): {len(all_chunks)}")

    if not all_chunks:
        print("HATA: Parçalanacak metin bulunamadı!")
        return
    
    print(f"'{MODEL_PATH}' yolundan model yükleniyor...")
    model = SentenceTransformer(str(MODEL_PATH))
    only_texts = [p['icerik'] for p in all_chunks]

    print("Metinler vektörleştiriliyor...")
    vectors = model.encode(only_texts, show_progress_bar=True, normalize_embeddings=True)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(VECTOR_DB_PATH))
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print("Veritabanı ve model başarıyla oluşturuldu.")

@app.on_event("startup")
def startup_event():
    print("Uygulama başlatılıyor...")
    load_dotenv()  # Load environment variables from .env file
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
        print("Gemini modeli başarıyla yapılandırıldı.")
    
    if not all(p.exists() for p in [MODEL_PATH, VECTOR_DB_PATH, METADATA_PATH]):
        create_database_and_models()

    print("Model ve veritabanı yükleniyor...")
    state["embedding_model"] = SentenceTransformer(str(MODEL_PATH))
    state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        state["chunks_metadata"] = json.load(f)
    print("Uygulama başarıyla başlatıldı!")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "UniAsistan API çalışıyor!"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not all(state.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uygulama henüz tam olarak başlatılmadı. Lütfen daha sonra tekrar deneyin."    
    )

    user_question_lower = request.question.lower().strip()

    # Meta sorular için özel cevaplar
    meta_questions_keywords = {
        ("sen kimsin", "kimsin sen", "nesin sen", "sen nesin"): "Ben Cihan Ayindi tarafından ADÜ Öğrencilerine Öğrenci işlerinde sorulabilecek konularda öğrencilere yardımcı olması amacıyla geliştirilmiş bir RAG sistemiyim.",
        ("kim geliştirdi", "geliştiricin kim", "seni kim yaptı"): "Ben Cihan Ayindi tarafından geliştirildim.",
        ("amacın ne", "ne işe yararsın", "görevin ne"): "Amacım, Adnan Menderes Üniversitesi öğrencilerine öğrenci işleriyle ilgili konularda yardımcı olmaktır."
        # İhtiyaç duyarsanız başka meta sorular ve cevaplar ekleyebilirsiniz.
    }

    for keywords, answer in meta_questions_keywords.items():
        if any(keyword in user_question_lower for keyword in keywords):
            # Bu meta soruya özel cevap ver, kaynak belirtme
            return AnswerResponse(answer=answer, sources=[]) # Kaynak olarak boş liste

    # Meta soru değilse, RAG sürecine devam et
    if not all(state.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uygulama henüz tam olarak başlatılmadı. Lütfen daha sonra tekrar deneyin."    
        )

    question_vector = state["embedding_model"].encode(
        [request.question], normalize_embeddings=True
    )
    distances, indices = state["faiss_index"].search(question_vector, 7)

    context_parts = [state["chunks_metadata"][i]['icerik'] for i in indices[0]]
    sources = {state["chunks_metadata"][i]['kaynak'] for i in indices[0]}
    context = "\n---\n".join(context_parts)

    # ----- DEBUG BAŞLANGIÇ -----
    print("\n--- GEMINI'YE GÖNDERİLEN BAĞLAM ---")
    for i, part in enumerate(context_parts):
        print(f"--- CHUNK {i+1} (Kaynak: {state['chunks_metadata'][indices[0][i]]['kaynak']}) ---")
        print(part[:500] + "..." if len(part) > 500 else part) # Chunk'ın ilk 500 karakteri
    print("--- BAĞLAM SONU ---\n")
    # ----- DEBUG SON -----

    prompt = f"""
    Sen Adnan Menderes Üniversitesi Öğrenci İşleri için bir yardım asistanısın.
    Görevin, sadece ve sadece aşağıda sana verdiğim 'Bağlam' metinlerini kullanarak kullanıcının sorusuna cevap vermektir.

    Kullanıcı bir "YÖNERGENİN AMACI"nı sorduğunda, bağlamda "Bu Yönergenin amacı", "Yönergenin amacı", "Amaç MADDE 1" gibi ifadelerle başlayan veya bir yönergenin genel amacını tanımlayan cümleleri bulmaya çalış. 
    Bağlamda bir konunun (örneğin bir eğitimin veya öğretimin) özel amacı da geçiyorsa, kullanıcı açıkça bunu sormadığı sürece, öncelikle YÖNERGENİN GENEL AMACINI belirt.
    Eğer cevap bağlamda açıkça yoksa veya emin değilsen, 'Bu konuda bilgi sahibi değilim.' de.
    Cevabın kesinlikle bu bağlamın dışına çıkmamalıdır.
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

