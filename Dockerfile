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

# Modeli önceden indirmek için gerekli kütüphaneleri yükleyin (huggingface_hub veya sentence_transformers)
# Eğer zaten requirements.txt içindeyse bu satıra gerek kalmaz
RUN .venv/bin/pip install huggingface_hub sentence-transformers

# main.py dosyanızdaki EMBEDDING_MODEL_NAME'e karşılık gelen model adını tanımlayın
ENV EMBEDDING_MODEL_NAME="paraphrase-multilingual-mpnet-base-v2"

# Modeli indirmek için bir Python betiği oluşturun ve çalıştırın
# Bu, modelin cache dizinine indirilmesini sağlar.
RUN <<EOF
.venv/bin/python -c "
from sentence_transformers import SentenceTransformer
import os

model_name = os.environ.get('EMBEDDING_MODEL_NAME')
if model_name:
    print(f'Downloading SentenceTransformer model: {model_name}')
    # Varsayılan Hugging Face cache konumunu kullanmak daha basittir:
    # /root/.cache/huggingface/hub veya HF_HOME ortam değişkeni tarafından belirtilen yer
    SentenceTransformer(model_name, trust_remote_code=True)
    print('Model download complete.')
else:
    print('EMBEDDING_MODEL_NAME environment variable not set. Skipping model download during build.')
"
EOF

# İkinci aşama: Minimal bir çalışma ortamı oluşturun
FROM python:3.13.2-slim

# Çalışma dizinini belirleyin
WORKDIR /app

# Sadece sanal ortamı ve indirilen modeli kopyalayın
COPY --from=builder /app/.venv .venv/

# Hugging Face cache dizinini de kopyalamamız gerekiyor
# Hugging Face varsayılan olarak /root/.cache/huggingface/hub içine indirir
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Geri kalan uygulama dosyalarını kopyalayın
COPY . .

# Uygulamayı başlatın
CMD [".venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
