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
from fastapi.responses import FileResponse
from typing import Optional, Set, Tuple
import logging
from logging.handlers import TimedRotatingFileHandler

log_handler = TimedRotatingFileHandler(
    "uniasistan.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding='utf-8'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        log_handler,
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# --- Constants and Settings ---
DATA_PATH = Path("./data") 
VECTOR_DB_PATH = DATA_PATH / "faiss_index.bin"
METADATA_PATH = DATA_PATH / "chunks_metadata.json"
SOURCE_DOCUMENTS_PATH = Path("./source_documents") 
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
GENERATIVE_MODEL_NAME = 'gemini-1.5-flash'
CATEGORIES_DATA_PATH = SOURCE_DOCUMENTS_PATH / "category.json"

# --- Pydantic Models ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

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
    "https://uniasistan.vercel.app", # Live deployment
    "http://localhost:8080", # For local development
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

# Startup event to load models and data 
def load_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("UYARI: GEMINI_API_KEY ortam değişkeni bulunamadı. API çalışmayabilir.")
        return
    genai.configure(api_key=api_key)
    state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    logger.info("Gemini modeli başarıyla yapılandırıldı.")

def load_embedding_model():
    logger.info(f"Embedding modeli '{EMBEDDING_MODEL_NAME}' yükleniyor...")
    state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_faiss_index():
    logger.info(f"FAISS index'i '{VECTOR_DB_PATH}' yolundan yükleniyor...")
    state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))

def load_metadata():
    logger.info(f"Metadata '{METADATA_PATH}' yolundan yükleniyor...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        state["chunks_metadata"] = json.load(f)

def check_source_documents():
    if not SOURCE_DOCUMENTS_PATH.is_dir():
        logger.info(f"UYARI: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        logger.info("PDF indirme ve listeleme özelliği çalışmayabilir.")
    else:
        logger.info(f"Kaynak dokümanlar klasörü bulundu: {SOURCE_DOCUMENTS_PATH}")

@app.on_event("startup")
def startup_event():
    """
    When the application starts, this function is called to load models and data.
    """
    logger.info("Uygulama başlatılıyor...")
    load_dotenv()

    load_gemini_model()

    try:
        load_embedding_model()
        load_faiss_index()
        load_metadata()
        logger.info("Uygulama başarıyla başlatıldı ve kullanıma hazır!")
    except FileNotFoundError as e:
        logger.info(f"KRİTİK HATA: Gerekli bir dosya bulunamadı: {e}")
        logger.info("Lütfen 'create_database_local.py' scriptini çalıştırdığınızdan ve 'data' klasörünün Docker imajına eklendiğinden emin olun.")
    except Exception as e:
        logger.info(f"Başlatma sırasında beklenmedik bir hata oluştu: {e}")

    check_source_documents()

# --- API Endpoints ---
@app.get("/")
async def root():
    """Main endpoint to check if the API is running."""
    return {"message": "UniAsistan API çalışıyor!"}

def check_models_loaded():
    """
    Its checks if all necessary models and data are loaded.
    If not, raises a 503 Service Unavailable error.
    """
    if not all([
        state["embedding_model"],
        state["faiss_index"],
        state["chunks_metadata"],
        state["generative_model"]
    ]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Uygulama henüz hazır değil. Lütfen logları kontrol edin."
        )
    
def is_it_meta_question(user_question: str) -> Optional[AnswerResponse]:
    """
    Its checks if the user's question is a meta question
    (e.g., "Sen kimsin?", "Kim geliştirdi?", "Amacın ne?").
    If it is, returns a predefined answer.
    """
    
    meta_questions_keywords = {
        ("sen kimsin", "kimsin sen", "nesin sen", "sen nesin"): 
            "Ben Cihan Ayindi tarafından ADÜ Öğrencilerine Öğrenci işlerine sorulabilecek konularda yardımcı olması amacıyla geliştirilmiş bir RAG sistemiyim.",
        ("kim geliştirdi", "geliştiricin kim", "seni kim yaptı"): 
            "Ben Cihan Ayindi tarafından geliştirildim.",
        ("amacın ne", "ne işe yararsın", "görevin ne"): 
            "Amacım, Adnan Menderes Üniversitesi öğrencilerine öğrenci işleriyle ilgili konularda yardımcı olmaktır. (Ve Cihan Ayindi'ye staj bulmasında yardımcı olmak :) )",
        # 1. Hal Hatır Sorma / Duygusal Durum
        ("nasılsın", "iyi misin", "keyfin nasıl", "ne var ne yok"):
            "Teşekkür ederim, iyiyim! Sana yardımcı olabildiğim sürece enerjim hep yüksek. Umarım sen de iyisindir. Hangi konuda bilgi almak istersin?",
        # 2. Yetenek ve Sınırlar
        ("neler yapabilirsin", "başka ne biliyorsun", "yeteneğin ne"):
            "Adnan Menderes Üniversitesi'nin yönetmelik, yönerge ve akademik kuralları hakkında sorularını cevaplayabilirim. Örneğin, 'staj başvurusu nasıl yapılır?' veya 'ders kayıt tarihleri ne zaman?' gibi sorular sorabilirsin.",
        ("ne yapamazsın", "bilmediğin ne var", "sınırların ne"):
            "Kişisel fikir belirtmem, yorum yapmam veya kişisel bilgilerine erişemem. Ayrıca, sistemimde yüklü olmayan veya güncel olmayan konular hakkında da bilgi veremem. Sadece bana öğretilen dokümanlar çerçevesinde cevaplar üretiyorum.",
        # 3. Teşekkür ve Vedalaşma
        ("teşekkür ederim", "teşekkürler", "sağ ol", "eyvallah"):
            "Rica ederim! Başka bir sorun olursa çekinme, yine beklerim.",
        ("görüşürüz", "hoşça kal", "bay bay"):
            "Görüşmek üzere! Başarılar dilerim.",
        # 4. Şaka ve Eğlence
        ("şaka yap", "fıkra anlat"):
            "Öğrenci işleri o kadar ciddi bir konu ki şaka yapmaya kodlarım izin vermiyor. :) Ama istersen sana bütünleme sınavı yönetmeliğini anlatabilirim, belki daha eğlencelidir?",
        # 5. Bilgi Kaynağı
        ("bilgileri nereden alıyorsun", "kaynağın ne", "emin misin"):
            "Cevaplarımı, geliştiricim tarafından bana sunulan Aydın Adnan Menderes Üniversitesi'nin resmi internet sitesinde yayınlanmış güncel yönetmelik ve yönerge PDF'lerinden alıyorum. Verdiğim cevapların altında kaynak dokümanları da listeliyorum.",
        # 6. Gerçek Zamanlı Bilgiler (Veremeyeceğini Belirtme)
        ("saat kaç", "bugün ayın kaçı", "hava nasıl"):
            "Benim saat, takvim veya hava durumu gibi gerçek zamanlı bilgilere erişimim yok. Sana sadece ADÜ öğrenci işleri dokümanları hakkında bilgi verebilirim."
    }

    for keywords, answer in meta_questions_keywords.items():
        if any(keyword in user_question for keyword in keywords):
            logger.info(f"Meta soru tespit edildi: {user_question}")
            return AnswerResponse(answer=answer, sources=[])

    return None

def get_context_parts(
    question: str,
    k: int = 7,
    allowed_sources: Optional[Set[str]] = None
    ) -> Tuple[str, Set[str]]:
    """
    For a given question, retrieves relevant context parts from the FAISS index.
    If allowed_sources is provided, filters the results to only include those sources.
    """

    question_vector = state["embedding_model"].encode(
        [question], normalize_embeddings=True
    ).astype("float32")

    # We use k * 3 to ensure we get enough results, as some may be filtered out
    distances, indices = state["faiss_index"].search(question_vector, k * 3)

    if not indices.size:
        return "Sorunuza uygun bir içerik bulamadım.", set()

    filtered_context_parts = []
    filtered_sources = set()

    for idx in indices[0]:
        chunk_meta = state["chunks_metadata"][idx]
        source = chunk_meta["kaynak"]

        # If allowed_sources is provided, filter by source
        if allowed_sources is not None and source not in allowed_sources:
            continue

        filtered_context_parts.append(chunk_meta["icerik"])
        filtered_sources.add(source)

        if len(filtered_context_parts) >= k:
            break

    if not filtered_context_parts:
        return "Sorunuza uygun bir içerik bulamadım.", set()

    return "\n---\n".join(filtered_context_parts), filtered_sources

def build_prompt_for_categorize(question: str) -> str:
    """
    Is a helper function to build the prompt for categorizing the user's question.
    """
    return f"""
    Aşağıdaki kullanıcı sorusu, hangi kategorilerle en çok ilişkilidir? Kategoriler aşağıda listelenmiştir. Lütfen yalnızca en uygun 3 kategoriyi sırasıyla belirt.

    Kullanıcı Sorusu: {question}

    Kategoriler:

    ---
    "Genel Akademik Süreçler ve Yönetmelikler": "Bu kategori, üniversitenin genel eğitim-öğretim yönetmeliklerini, akademik takvimi, öğrenim süreleri ve genel işleyişle ilgili temel kuralları içerir. Tüm lisans ve ön lisans öğrencilerini ilgilendiren temel düzenlemeler burada bulunur.",
    "Ders, Kayıt ve Muafiyet İşlemleri": "Ders seçimi, ders kaydı, ders saydırma (muafiyet) ve intibak işlemleri gibi konulara ilişkin yönetmelik ve esasları kapsar. Öğrencilerin akademik dönem başında yapması gereken kayıt işlemleri ve ders denklikleri bu kategoridedir.",
    "Sınavlar ve Değerlendirme": "Vize, final, mazeret ve bütünleme gibi sınavların uygulanma usullerini, değerlendirme kriterlerini ve sınav kurallarını içerir. Sınavlara dair tüm yönetmelikler, esaslar ve notlandırma ile ilgili belgeler bu başlık altındadır.",
    "Mezuniyet ve Diploma İşlemleri": "Mezuniyet koşulları, diploma, diploma eki ve diğer belgelerin düzenlenmesi, ilgili ücretler ve ilişik kesme gibi süreçleri kapsar. Öğrenimini tamamlayan öğrencilerin yapması gereken işlemlerle ilgili tüm bilgiler burada yer alır.",
    "Yatay Geçiş, Çift Anadal ve Özel Öğrenci Statüleri": "Kurum içi veya kurumlar arası yatay geçiş, çift anadal (ÇAP), yandal ve özel öğrenci statüsüne başvuru koşulları ve süreçleri hakkında bilgi verir. Farklı programlar veya üniversiteler arası geçiş yapmak isteyen öğrenciler için ilgili yönergeler buradadır.",
    "Staj ve Uygulamalı Eğitimler": "Bu kategori, öğrencilerin zorunlu veya isteğe bağlı stajları, iş yeri eğitimleri ve diğer uygulamalı derslerle ilgili yönerge, genel kurallar, amaçlar, süreçleri, gerekli belgeleri içerir. Staj başvurusu, defter hazırlama, değerlendirme ve staj yönergelerinin hedefleri gibi konular bu kategoride yer alır.",
    "Yabancı Dil ve Yabancı Uyruklu Öğrenci İşlemleri": "Yabancı dil hazırlık sınıfları, muafiyet sınavları ve yabancı uyruklu öğrencilerin üniversiteye kabul süreçleri, başvuru şartları ve kayıt işlemleri ile ilgili tüm düzenlemeleri içerir.",
    "Lisansüstü Eğitim (Enstitüler)": "Yüksek lisans ve doktora programları ile ilgilidir. Başvuru, kayıt, ders, tez, uzmanlık alan dersi ve yeterlilik sınavı gibi lisansüstü eğitim süreçlerine dair yönetmelik ve esasları kapsar.",
    "Öğrenci Yaşamı, Hakları ve Destek Birimleri": "Öğrenci disiplin yönetmeliği, öğrenci konseyi, öğrenci danışmanlığı (mentorlük), özel gereksinimli veya engelli öğrencilere sunulan destekler ve fırsat eşitliği gibi konuları içerir.",
    "Fakülte ve Yüksekokul Özel Yönetmelikleri": "Sadece belirli bir fakülte, enstitü veya yüksekokula özgü eğitim-öğretim, sınav, staj veya işleyiş kurallarını barındırır. Mühendislik, Tıp, Diş Hekimliği gibi fakültelerin kendi iç yönergeleri bu kategoridedir.",
    "Formasyon ve Sertifika Programları": "Öğretmenlik için gerekli olan pedagojik formasyon eğitimi gibi özel sertifika programlarına ilişkin başvuru, eğitim ve değerlendirme esaslarını içerir.",
    "İdari ve Organizasyonel Belgeler": "Üniversitenin akademik ve idari teşkilat yapısı, komisyonların çalışma usulleri, stratejik raporlar ve yaz okulu gibi genel işleyişe dair kurumsal belgeleri içerir. Bu kategori doğrudan öğrenci işlemlerinden çok, kurumun iç işleyişi ile ilgilidir."
    ---

    Cevabınızı yalnızca aşağıdaki kategori başlıklarından bir listeyle ve aynen yazıldığı şekilde veriniz. Ek açıklama yapmayınız.
    
    Cevap formatı:
    "Kategori 1", "Kategori 2", "Kategori 3"
    """

def send_api_request_for_categorize(question: str):
    """
    It sends a request to the Gemini API to categorize the user's question.
    """
    
    try:
        response = state["generative_model"].generate_content(build_prompt_for_categorize(question))
        logger.info(f"Gemini API kategorize cevabı: {response.text.strip()}")
        return response.text
    except Exception as e:
        logger.info(f"Gemini API kategorize hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Soru kategorize edilirken bir hata oluştu."
        )

def take_filenames_from_sources(categories: list[str], json_path: str) -> set[str]:
    """
    It retrieves the file names associated with the given categories
    from the category file map stored in a JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        category_file_map = json.load(f)

    selected_files = set()

    for category in categories:
        files = category_file_map.get(category, [])
        selected_files.update(files)

    return selected_files

def build_prompt(question: str, context: str) -> str:
    """
    It builds the prompt for the Gemini API based on the user's question and the context.
    The context is the relevant information retrieved from the FAISS index.
    """

    logger.info(f"'{question}' sorusu için bulunan bağlam:\n{context}") 

    return f"""
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
    Kullanıcı Sorusu: {question}
    Cevap:"""

def send_api_request(prompt: str):
    """
    It sends a request to the Gemini API to generate an answer based on the user's question and context."""
    try:
        response = state["generative_model"].generate_content(prompt)
        return response.text
    except Exception as e:
        logger.info(f"Gemini API hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cevap üretilirken bir hata oluştu."
        )

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    It handles the user's question, checks if models are loaded,
    categorizes the question, retrieves relevant context,
    builds the prompt, and sends it to the Gemini API to get an answer.
    """
    check_models_loaded()  # Check if models are loaded
    
    user_question_lower = request.question.lower().strip() # Change the question to lowercase and strip whitespace

    meta_answer = is_it_meta_question(user_question_lower) # Check if the question is a meta question
    if meta_answer:
        return meta_answer  # If it is a meta question, return the predefined answer
    
    categories_raw = send_api_request_for_categorize(request.question).split(",") # Categorize the question using Gemini API
    categories = [cat.strip().strip('"') for cat in categories_raw]

    allowed_files = take_filenames_from_sources(categories, CATEGORIES_DATA_PATH) # Take the file names associated with the categories from the category file map

    context, sources_rag = get_context_parts(request.question,k=7,allowed_sources=allowed_files) # Take the relevant context parts from the FAISS index

    prompt = build_prompt(request.question, context) # Build the prompt for the Gemini API

    return AnswerResponse(answer = send_api_request(prompt), sources=list(sources_rag)) # Send the prompt to the Gemini API and return the answer and sources

@app.get("/get_sourceslist", response_model=list[SourceFile])
async def get_sources_list():
    """
    It retrieves the list of PDF files in the source_documents directory.
    Returns a list of dictionaries with file names and URLs for downloading.
    """
    if not SOURCE_DOCUMENTS_PATH.is_dir(): # Check if the source_documents directory exists
        logger.info(f"Hata: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        return []

    pdf_files = []
    for item in SOURCE_DOCUMENTS_PATH.iterdir(): # Iterate through the items in the source_documents directory
        if item.is_file() and item.suffix.lower() == ".pdf":
            pdf_files.append({
                "name": item.name,
                "url": f"/download_source/{item.name}" # Create the URL for downloading the PDF file
            })
    
    if not pdf_files: # If no PDF files are found, log a message
        logger.info(f"Bilgi: {SOURCE_DOCUMENTS_PATH} klasöründe PDF dosyası bulunamadı.") 

    return pdf_files


@app.get("/download_source/{filename:path}")
async def download_source_file(filename: str):
    """
    It handles the file download request for a specific PDF file.
    Validates the filename, checks if the file exists, 
    and returns the file response for download.
    Raises HTTP exceptions for invalid filename or file not found.
    """
    try:
        safe_filename = os.path.basename(filename) # Get the base name of the file to prevent directory traversal attacks
        if safe_filename != filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Geçersiz dosya adı.")

        file_path = SOURCE_DOCUMENTS_PATH / safe_filename

        if not file_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dosya bulunamadı.")
        
        return FileResponse(
            path=str(file_path), 
            media_type='application/pdf',
            filename=safe_filename 
        )
    except HTTPException:
        raise 
    except Exception as e:
        print(f"Dosya indirme hatası ({filename}): {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Dosya indirilirken bir sunucu hatası oluştu.")