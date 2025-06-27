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
from typing import Optional, Set, Tuple

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

def load_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("UYARI: GEMINI_API_KEY ortam değişkeni bulunamadı. API çalışmayabilir.")
        return
    genai.configure(api_key=api_key)
    state["generative_model"] = genai.GenerativeModel(GENERATIVE_MODEL_NAME)
    print("Gemini modeli başarıyla yapılandırıldı.")

def load_embedding_model():
    print(f"Embedding modeli '{EMBEDDING_MODEL_NAME}' yükleniyor...")
    state["embedding_model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)

def load_faiss_index():
    print(f"FAISS index'i '{VECTOR_DB_PATH}' yolundan yükleniyor...")
    state["faiss_index"] = faiss.read_index(str(VECTOR_DB_PATH))

def load_metadata():
    print(f"Metadata '{METADATA_PATH}' yolundan yükleniyor...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        state["chunks_metadata"] = json.load(f)

def check_source_documents():
    if not SOURCE_DOCUMENTS_PATH.is_dir():
        print(f"UYARI: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        print("PDF indirme ve listeleme özelliği çalışmayabilir.")
    else:
        print(f"Kaynak dokümanlar klasörü bulundu: {SOURCE_DOCUMENTS_PATH}")

@app.on_event("startup")
def startup_event():
    print("Uygulama başlatılıyor...")
    load_dotenv()

    load_gemini_model()

    try:
        load_embedding_model()
        load_faiss_index()
        load_metadata()
        print("Uygulama başarıyla başlatıldı ve kullanıma hazır!")
    except FileNotFoundError as e:
        print(f"KRİTİK HATA: Gerekli bir dosya bulunamadı: {e}")
        print("Lütfen 'create_database_local.py' scriptini çalıştırdığınızdan ve 'data' klasörünün Docker imajına eklendiğinden emin olun.")
    except Exception as e:
        print(f"Başlatma sırasında beklenmedik bir hata oluştu: {e}")

    check_source_documents()

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "UniAsistan API çalışıyor!"}

def check_models_loaded():
    """
    Modellerin ve veritabanının yüklü olup olmadığını kontrol eder.
    Eğer yüklenmemişse, HTTP 503 hatası fırlatır.
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
    Kullanıcının sorusunun meta bir soru olup olmadığını kontrol eder.
    Meta sorular, sistemin kendisi veya amacıyla ilgili sorulardır.
    """
    meta_questions_keywords = {
        ("sen kimsin", "kimsin sen", "nesin sen", "sen nesin"): 
            "Ben Cihan Ayindi tarafından ADÜ Öğrencilerine Öğrenci işlerinde sorulabilecek konularda öğrencilere yardımcı olması amacıyla geliştirilmiş bir RAG sistemiyim.",
        ("kim geliştirdi", "geliştiricin kim", "seni kim yaptı"): 
            "Ben Cihan Ayindi tarafından geliştirildim.",
        ("amacın ne", "ne işe yararsın", "görevin ne"): 
            "Amacım, Adnan Menderes Üniversitesi öğrencilerine öğrenci işleriyle ilgili konularda yardımcı olmaktır. (Ve Cihan Ayindi'ye staj bulmasında yardımcı olmak :) )",
    }

    for keywords, answer in meta_questions_keywords.items():
        if any(keyword in user_question for keyword in keywords):
            print(f"Meta soru tespit edildi: {user_question}")
            return AnswerResponse(answer=answer, sources=[])

    return None

def get_context_parts(
    question: str,
    k: int = 7,
    allowed_sources: Optional[Set[str]] = None
) -> Tuple[str, Set[str]]:
    """
    Soru için embedding hesaplar, FAISS üzerinden arama yapar,
    sadece allowed_sources içinde olan chunk'lardan k adet içerik döner.

    Args:
        question (str): Kullanıcının sorduğu soru.
        k (int): Döndürülecek içerik parça sayısı.
        allowed_sources (Optional[Set[str]]): İzin verilen kaynak dosya isimleri.

    Returns:
        Tuple[str, Set[str]]: Birleşik içerik metni ve içeriklerin kaynak dosya isimleri.
    """

    question_vector = state["embedding_model"].encode(
        [question], normalize_embeddings=True
    ).astype("float32")

    # Daha fazla chunk çekiyoruz ki filtre sonrası yeterli sayıda kalabilsin
    distances, indices = state["faiss_index"].search(question_vector, k * 3)

    if not indices.size:
        return "Sorunuza uygun bir içerik bulamadım.", set()

    filtered_context_parts = []
    filtered_sources = set()

    for idx in indices[0]:
        chunk_meta = state["chunks_metadata"][idx]
        source = chunk_meta["kaynak"]

        # Eğer allowed_sources tanımlıysa ve kaynak izin verilenlerde değilse atla
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
    Kullanıcının sorusunu kategorize etmek için kullanılacak prompt'u oluşturur.
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
    Google Gemini API'sine soru kategorize etme isteği gönderir.
    """
    
    try:
        response = state["generative_model"].generate_content(build_prompt_for_categorize(question))
        return response.text
    except Exception as e:
        print(f"Gemini API kategorize hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Soru kategorize edilirken bir hata oluştu."
        )

def take_filenames_from_sources(categories: list[str], json_path: str) -> set[str]:
    """
    Kategorilere ait dosya adlarını category.json'dan alır.

    Args:
        categories (list[str]): Seçilen kategori isimleri
        json_path (str): category.json dosyasının yolu

    Returns:
        set[str]: İlgili kategoriye ait dosya adları
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
    Kullanıcının sorusuna cevap vermek için kullanılacak prompt'u oluşturur.
    """
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
    Google Gemini API'sine istek gönderir ve cevabı döndürür.
    """
    try:
        response = state["generative_model"].generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cevap üretilirken bir hata oluştu."
        )

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    check_models_loaded()  # Modellerin yüklü olup olmadığını kontrol et 
    
    user_question_lower = request.question.lower().strip() # Soru metnini küçük harfe çevir ve baştaki/sondaki boşlukları temizle

    meta_answer = is_it_meta_question(user_question_lower) # Meta bir soru mu kontrol et
    if meta_answer:
        return meta_answer  # Eğer meta bir soruysa, direkt cevabı dön
    
    categories_raw = send_api_request_for_categorize(request.question).split(",")
    categories = [cat.strip().strip('"') for cat in categories_raw]

    allowed_files = take_filenames_from_sources(categories, CATEGORIES_DATA_PATH) # Kategorilere göre dosya adlarını al

    print(f"Kategoriler: {categories}, İzin verilen dosyalar: {allowed_files}")

    context, sources_rag = get_context_parts(request.question,k=7,allowed_sources=allowed_files) # Cevap için bağlamı ve kaynakları al

    prompt = build_prompt(request.question, context) # Prompt'u oluştur

    return AnswerResponse(answer = send_api_request(prompt), sources=list(sources_rag))

@app.get("/get_sourceslist", response_model=list[SourceFile])
async def get_sources_list():
    """
    source_documents klasöründeki PDF dosyalarını listeler.
    Her dosya için adını ve indirme URL'sini döndürür.
    """
    if not SOURCE_DOCUMENTS_PATH.is_dir(): # Kaynak dokümanlar klasörü kontrolü
        print(f"Hata: Kaynak dokümanlar klasörü bulunamadı: {SOURCE_DOCUMENTS_PATH}")
        return []

    pdf_files = []
    for item in SOURCE_DOCUMENTS_PATH.iterdir(): # Klasördeki her dosyayı kontrol et
        if item.is_file() and item.suffix.lower() == ".pdf":
            pdf_files.append({
                "name": item.name,
                "url": f"/download_source/{item.name}" # Frontend'in beklediği URL formatı
            })
    
    if not pdf_files: # Eğer klasörde PDF dosyası yoksa
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