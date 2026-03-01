import os
import json
import re
import numpy as np
import torch
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from razdel import sentenize
import pymorphy3
from collections import Counter

# Импорт модуля чтения файлов
from file_reader import read_file, get_supported_files, extract_metadata

# ==========================
# НАСТРОЙКИ
# ==========================

DATA_DIR = "data"
MODEL_NAME = "intfloat/multilingual-e5-large"
BATCH_SIZE = 64
SIM_THRESHOLD = 0.80
MIN_SPECIES_OCC = 3

# OCR настройки (EasyOCR)
USE_OCR = True  # Включить OCR для сканированных PDF/DJVU
OCR_LANG = ["ru", "en"]  # Языки распознавания
USE_GPU_OCR = True  # Использовать GPU для OCR (если доступен)
DPI_SCALE = 3.0  # Масштаб рендера (3.0 = ~300 DPI)
MIN_TEXT_RATIO = 0.10  # ✅ Мин. доля текста для пропуска OCR
OCR_PAGE_LIMIT = 500  # ✅ Лимит страниц для OCR на файл

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA доступна:", torch.cuda.is_available())
print("Используемое устройство:", device)

# ==========================
# МОДЕЛИ
# ==========================

embedder = SentenceTransformer(MODEL_NAME, device=device)
if device == "cuda":
    embedder.half()

morph = pymorphy3.MorphAnalyzer()


# ==========================
# ФУНКЦИИ ОБРАБОТКИ ТЕКСТА
# ==========================

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()



def lemmatize_phrase(text):
    words = re.findall(r"[А-Яа-яЁёA-Za-z\-]+", text)
    return " ".join(morph.parse(w)[0].normal_form for w in words)


def extract_candidate_species(text):
    pattern = r'\b[А-ЯЁ][а-яё]+(?:\s[а-яё]+){0,3}'
    return re.findall(pattern, text)


def semantic_chunk(text):
    sentences = [s.text for s in sentenize(text)]
    if len(sentences) < 2:
        return sentences

    sent_emb = embedder.encode(
        ["passage: " + s for s in sentences],
        normalize_embeddings=True,
        batch_size=128
    )

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = np.dot(sent_emb[i - 1], sent_emb[i])
        if sim < SIM_THRESHOLD:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks


# ==========================
# ОБРАБОТКА ФАЙЛОВ
# ==========================

all_chunks = []
metadata = []
species_counter = Counter()

files = get_supported_files(DATA_DIR)
print(f"📁 Найдено файлов: {len(files)}")
print(f"   Форматы: {set(os.path.splitext(f)[1] for f in files)}")

for file in tqdm(files, desc="Файлы"):
    file_path = os.path.join(DATA_DIR, file)
    file_ext = os.path.splitext(file)[1].lower()

    # Чтение файла с OCR поддержкой
    raw_text = read_file(
        file_path,
        use_ocr=USE_OCR,
        ocr_lang=OCR_LANG,
        gpu=USE_GPU_OCR,
        dpi_scale=DPI_SCALE,
        min_text_ratio=MIN_TEXT_RATIO,  # ✅ Добавьте это
        ocr_page_limit=OCR_PAGE_LIMIT   # ✅ и это
    )

    if not raw_text or len(raw_text.strip()) < 50:
        print(f"   ⚠️ Пропущен: пустой или нечитаемый")
        continue

    # Извлечение метаданных
    file_meta = extract_metadata(file_path)

    # Очистка и нормализация
    text = normalize_text(raw_text)

    # Извлечение видов растений
    candidates = extract_candidate_species(text)
    for c in candidates:
        lemma = lemmatize_phrase(c)
        species_counter[lemma] += 1

    # Семантическое чанкование
    chunks = semantic_chunk(text)

    # Сохранение чанков с метаданными
    for chunk in chunks:
        all_chunks.append("passage: " + chunk)
        metadata.append({
            "source": file,
            "format": file_ext.lstrip("."),
            "title": file_meta.get("title"),
            "author": file_meta.get("author"),
            "length": len(chunk)
        })

    print(f"   ✅ {file}: {len(chunks)} чанков, {len(candidates)} видов")

print(f"\n📊 Итого чанков: {len(all_chunks)}")
print(f"📊 Уникальных видов: {len(species_counter)}")

# ==========================
# ВИДЫ
# ==========================

species_final = [
    sp for sp, count in species_counter.items()
    if count >= MIN_SPECIES_OCC
]

with open("species_list.json", "w", encoding="utf-8") as f:
    json.dump(species_final, f, ensure_ascii=False, indent=2)

print(f"🌱 Сохранено видов (≥{MIN_SPECIES_OCC} вхождений): {len(species_final)}")

# ==========================
# EMBEDDINGS
# ==========================

print("\n🔢 Создание эмбеддингов...")
embeddings = embedder.encode(
    all_chunks,
    batch_size=BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

# ==========================
# FAISS INDEX (CPU)
# ==========================

dimension = embeddings.shape[1]
n_vectors = embeddings.shape[0]

print(f"\n📐 Векторов: {n_vectors}, Размерность: {dimension}")

if n_vectors < 5000:
    index = faiss.IndexFlatIP(dimension)
    print("✅ Индекс: IndexFlatIP (точный поиск)")
else:
    nlist = int(np.sqrt(n_vectors))
    nlist = max(32, min(nlist, 512))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    print(f"✅ Индекс: IndexIVFFlat (nlist={nlist})")

index.add(embeddings)
faiss.write_index(index, "plants.index")
print("💾 Индекс сохранён: plants.index")

# ==========================
# СОХРАНЕНИЕ ДАННЫХ
# ==========================

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("💾 Данные сохранены: chunks.json, metadata.json")

# ==========================
# ТЕСТ ПОИСКА
# ==========================

print("\n🔍 Тест поиска:")
test_queries = [
    "passage: роза садовая",
    "passage: помидоры уход",
    "passage: деревья плодовые"
]

for query in test_queries:
    q_emb = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb, 3)

    print(f"\nЗапрос: {query.replace('passage: ', '')}")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx < len(all_chunks):
            text = all_chunks[idx].replace("passage: ", "")[:120]
            source = metadata[idx].get("source", "unknown")
            fmt = metadata[idx].get("format", "txt")
            print(f"  {i + 1}. {dist:.4f} | [{fmt}] {source}: {text}...")

print("\n✅ INDEX READY 🚀")
