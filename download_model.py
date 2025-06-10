from sentence_transformers import SentenceTransformer
import os

# İndirilecek modelin adı
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'
# Modelin kaydedileceği klasör
MODEL_PATH = 'sbert_model'

# Klasör yoksa oluştur
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

print(f"'{MODEL_NAME}' modeli indiriliyor ve '{MODEL_PATH}' klasörüne kaydediliyor...")

# Modeli indir ve belirtilen yola kaydet
model = SentenceTransformer(MODEL_NAME)
model.save(MODEL_PATH)

print(f"Model başarıyla '{MODEL_PATH}' klasörüne kaydedildi.")