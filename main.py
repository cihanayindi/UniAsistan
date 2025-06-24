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
from fastapi.responses import FileResponse # Dosya indirme için eklendi
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# --- Constants and Settings ---
# Bu yollar, Docker imajı içindeki uygulama kök dizinine göre belirlenmiştir.
DATA_PATH = Path("./data") 
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
SOURCE_DOCUMENTS_PATH = Path("./source_documents") # Yeni: Kaynak dokümanların yolu
# Bu model artık lokalden değil, doğrudan kütüphane üzerinden yüklenecek.
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

# Yeni Pydantic Model (isteğe bağlı ama iyi pratik)
class SourceFile(BaseModel):
    name: str
    url: str

# --- FastAPI App ---
app = FastAPI(
    title="UniAsistan API",
    description="Adnan Menderes Üniversitesi için RAG tabanlı yardım asistanı.",
    version="1.0.0"
)

# CORS Middleware
origins = [
    "https://uniasistan.vercel.app",
    "http://localhost:8080", # Frontend development için
    "http://127.0.0.1:8080", # Frontend development için
    "null", # Lokal HTML dosyalarını açarken (file://) bazen origin null olur
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
    except Exception as e:
        print(f"Başlatma sırasında beklenmedik bir hata oluştu: {e}")

    # Kaynak dokümanlar klasörünün varlığını kontrol et
    if not SOURCE_DOCUMENTS_PATH.is_dir():
        print(f"UYARI: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        print("PDF indirme ve listeleme özelliği çalışmayabilir.")
    else:
        print(f"Kaynak dokümanlar klasörü bulundu: {SOURCE_DOCUMENTS_PATH}")


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "UniAsistan API çalışıyor!"}

# 1. YARDIMCI FONKSİYON (Bu fonksiyonu @app.post'tan önceye ekleyin)
def reorder_with_mmr(
    question_vector: np.ndarray,
    faiss_index: faiss.Index,
    initial_indices: np.ndarray,
    k: int,
    lambda_mult: float = 0.7
) -> list[int]:
    """
    FAISS'ten gelen sonuçları MMR (Maximal Marginal Relevance) kullanarak yeniden sıralar.
    """
    if initial_indices.size == 0:
        return []

    # --- DÜZELTME BU SATIRDA ---
    # 'i' değişkenini standart bir Python tam sayısına (int(i)) dönüştürüyoruz.
    # Ayrıca güvenlik için -1 olan indisleri de filtreliyoruz.
    valid_indices = [i for i in initial_indices if i != -1]
    retrieved_vectors = np.array([faiss_index.reconstruct(int(i)) for i in valid_indices])
    # --------------------------

    if retrieved_vectors.size == 0:
        return []

    # Soru vektörü ile alınan tüm belgeler arasındaki benzerliği hesapla
    query_similarity = cosine_similarity(question_vector, retrieved_vectors)[0]

    # Alınan belgelerin kendi aralarındaki benzerliğini hesapla
    doc_similarity = cosine_similarity(retrieved_vectors)

    final_indices = []
    remaining_indices_positions = list(range(len(valid_indices)))
    
    most_relevant_pos = np.argmax(query_similarity)
    final_indices.append(valid_indices[most_relevant_pos])
    remaining_indices_positions.pop(most_relevant_pos)

    while len(final_indices) < min(k, len(valid_indices)):
        mmr_scores = {}
        for pos in remaining_indices_positions:
            # Not: doc_similarity matrisinde doğru satır ve sütunları kullanıyoruz.
            selected_positions = [valid_indices.index(i) for i in final_indices]
            similarity_to_selected = np.max(doc_similarity[pos, selected_positions])
            
            mmr_score = lambda_mult * query_similarity[pos] - (1 - lambda_mult) * similarity_to_selected
            mmr_scores[pos] = mmr_score
        
        best_pos = max(mmr_scores, key=mmr_scores.get)
        final_indices.append(valid_indices[best_pos])
        remaining_indices_positions.remove(best_pos)
        
    return final_indices


# 2. ANA FONKSİYON (Mevcut /ask endpoint'inizle bunu değiştirin)
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

    # --- KAPSAMLI HATA YAKALAMA VE MMR SÜRECİ ---
    try:
        # 1. Soruyu vektöre çevir
        question_vector = state["embedding_model"].encode(
            [request.question], normalize_embeddings=True
        ).astype(np.float32)
        
        # 2. MMR için daha geniş bir arama yap (performans için fetch_k'yı ayarlayabilirsiniz)
        fetch_k = 28
        distances, initial_indices = state["faiss_index"].search(question_vector, fetch_k)

        if initial_indices.size == 0:
            return AnswerResponse(answer="Sorunuza uygun bir içerik bulamadım.", sources=[])
        
        initial_indices = initial_indices[0]

        # 3. Sonuçları MMR ile yeniden sırala (sonuç sayısını final_k ile belirle)
        final_k = 7
        mmr_indices = reorder_with_mmr(
            question_vector=question_vector,
            faiss_index=state["faiss_index"],
            initial_indices=initial_indices,
            k=final_k
        )

        # 4. Bağlam ve kaynakları oluştur
        context_parts = [state["chunks_metadata"][i]['icerik'] for i in mmr_indices]
        sources_rag = {state["chunks_metadata"][i]['kaynak'] for i in mmr_indices}
        context = "\n---\n".join(context_parts)

        # 5. Prompt'u hazırla
        prompt = f"""
        Sen, Adnan Menderes Üniversitesi Öğrenci İşleri için çalışan bir yardım asistanısın. Görevin, yalnızca aşağıda verilen 'Bağlam' metnine dayanarak kullanıcının sorusuna cevap vermektir. 

        Aşağıdaki kurallara uymalısın:
        - Yanıtlarında sadece 'Bağlam' içeriğini kullan.
        - Bağlam dışı bilgi vermekten kaçın.
        - Cevaplarını kısa, anlaşılır ve nazik bir dille yaz.
        - Eğer Bağlam'da doğrudan veya dolaylı olarak cevabı çıkarabileceğin bilgi yoksa, sadece şu cevabı ver: "Bu konuda bilgi sahibi değilim. Bilgi sahibi olduğum belgeleri incelemek istersen kaynaklar sayfasına göz atabilirsin."

        ---
        Bağlam:
        {context}
        ---
        Kullanıcı Sorusu: {request.question}
        Cevap:"""
        
        # 6. Üretici modele gönder ve cevabı al
        response = state["generative_model"].generate_content(prompt)
        answer = response.text

        return AnswerResponse(answer=answer, sources=list(sources_rag))

    except Exception as e:
        # Hata durumunda, hatanın detaylarını sunucu loglarına yazdır.
        # Bu sayede "Failed to fetch" hatasının gerçek nedenini görebilirsin.
        print(f"!!! /ask ENDPOINT'İNDE KRİTİK HATA !!!")
        traceback.print_exc()

        # Kullanıcıya genel bir sunucu hatası mesajı döndür.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sunucuda bir hata oluştu. Lütfen logları kontrol edin. Hata: {e}"
        )


# --- YENİ ENDPOINT'LER ---

@app.get("/get_sourceslist", response_model=list[SourceFile])
async def get_sources_list():
    """
    source_documents klasöründeki PDF dosyalarını listeler.
    Her dosya için adını ve indirme URL'sini döndürür.
    """
    if not SOURCE_DOCUMENTS_PATH.is_dir():
        print(f"Hata: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        # Frontend'e boş liste döndürmek genellikle daha iyi bir kullanıcı deneyimi sağlar
        # Veya bir HTTPException fırlatabilirsiniz:
        # raise HTTPException(status_code=404, detail="Kaynak dokümanlar klasörü bulunamadı.")
        return []

    pdf_files = []
    for item in SOURCE_DOCUMENTS_PATH.iterdir():
        if item.is_file() and item.suffix.lower() == ".pdf":
            pdf_files.append({
                "name": item.name,
                "url": f"/download_source/{item.name}" # Frontend'in beklediği URL formatı
            })
    
    if not pdf_files:
        print(f"Bilgi: {SOURCE_DOCUMENTS_PATH} klasöründe PDF dosyası bulunamadı.")

    return pdf_files


@app.get("/download_source/{filename:path}")
async def download_source_file(filename: str):
    """
    Belirtilen dosyayı source_documents klasöründen indirir.
    {filename:path} kullanmak, dosya adında / gibi karakterler olsa bile (pek olası olmasa da)
    doğru çalışmasını sağlar. Genelde sadece {filename} yeterlidir.
    """
    try:
        # Güvenlik önlemi: Path traversal saldırılarını önlemek için.
        # Dosya adının sadece dosya adı olduğundan ve ../ gibi şeyler içermediğinden emin olalım.
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz dosya adı.")

        file_path = SOURCE_DOCUMENTS_PATH / safe_filename

        if not file_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dosya bulunamadı.")
        
        # Dosyayı indirme olarak sun
        return FileResponse(
            path=str(file_path), 
            media_type='application/pdf',
            filename=safe_filename # Bu, Content-Disposition: attachment; filename="dosya_adi.pdf" başlığını ayarlar
        )
    except HTTPException:
        raise # FastAPI tarafından zaten oluşturulmuş HTTP hatalarını tekrar fırlat
    except Exception as e:
        print(f"Dosya indirme hatası ({filename}): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Dosya indirilirken bir sunucu hatası oluştu.")