# İlk aşama: Bağımlılıkları yükleyin ve modeli indirin
FROM python:3.13.2 AS builder

# Ortam değişkenlerini ayarlayın
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Çalışma dizinini belirleyin
WORKDIR /app

# Python sanal ortamını oluşturun
RUN python -m venv .venv

# requirements.txt dosyasını kopyalayın ve bağımlılıkları yükleyin
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# Modeli önceden indirmek için gerekli kütüphaneleri yükleyin (huggingface_hub ve sentence_transformers)
RUN .venv/bin/pip install huggingface_hub sentence-transformers

# main.py dosyanızdaki EMBEDDING_MODEL_NAME'e karşılık gelen model adını tanımlayın
ENV EMBEDDING_MODEL_NAME="paraphrase-multilingual-mpnet-base-v2"
# Modelin kaydedileceği yolu tanımlayın (main.py'deki yeni MODEL_PATH ile uyumlu)
ENV MODEL_SAVE_PATH="/app/models/sbert_model"

# Modelin kaydedileceği dizini oluşturun
RUN mkdir -p ${MODEL_SAVE_PATH}

# Modeli indirmek ve belirlenen yola kaydetmek için bir Python betiği oluşturun ve çalıştırın
RUN <<EOF
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
import os

model_name = os.environ.get('EMBEDDING_MODEL_NAME')
model_save_path = os.environ.get('MODEL_SAVE_PATH')

if model_name and model_save_path:
    print(f'Downloading SentenceTransformer model: {model_name} and saving to {model_save_path}')
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model.save(model_save_path)
    print('Model download and save complete.')
else:
    print('EMBEDDING_MODEL_NAME or MODEL_SAVE_PATH environment variable not set. Skipping model download during build.')
"
EOF

# İkinci aşama: Minimal bir çalışma ortamı oluşturun
FROM python:3.13.2-slim

# Çalışma dizinini belirleyin
WORKDIR /app

# Sadece sanal ortamı kopyalayın
COPY --from=builder /app/.venv .venv/

# İndirilmiş ve kaydedilmiş modeli kopyalayın
COPY --from=builder /app/models/sbert_model /app/models/sbert_model

# source_documents klasörünü (PDF'lerinizin olduğu yer) kopyalayın
COPY ./source_documents /app/source_documents

# Geri kalan uygulama dosyalarını kopyalayın
COPY . .

# Uygulamayı başlatın
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
