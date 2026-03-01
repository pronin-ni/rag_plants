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
import argparse

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

# OCR настройки
USE_OCR = True
OCR_LANG = ["ru", "en"]
USE_GPU_OCR = True
DPI_SCALE = 3.0
MIN_TEXT_RATIO = 0.10
OCR_PAGE_LIMIT = 1000

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
# CLI + RESUME / CHECKPOINT
# ==========================
parser = argparse.ArgumentParser(description="Предобработка для Plant RAG")
parser.add_argument("--force", action="store_true", help="Пересчитать всё с нуля")
args = parser.parse_args()

CHUNKS_PATH = "chunks.json"
METADATA_PATH = "metadata.json"
SPECIES_PATH = "species_list.json"

resume = (os.path.exists(CHUNKS_PATH) and
          os.path.exists(METADATA_PATH) and
          os.path.exists(SPECIES_PATH) and not args.force)

if resume:
    print("✅ Режим resume: загружаем готовые чанки, метаданные и список видов")
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        all_chunks = json.load(f)
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    with open(SPECIES_PATH, encoding="utf-8") as f:
        species_final = json.load(f)
    print(f"📊 Загружено: {len(all_chunks)} чанков, {len(species_final)} видов")
else:
    all_chunks = []
    metadata = []
    species_counter = Counter()
    files = get_supported_files(DATA_DIR)
    print(f"📁 Найдено файлов: {len(files)}")
    print(f" Форматы: {set(os.path.splitext(f)[1] for f in files)}")

    for file in tqdm(files, desc="Файлы"):
        file_path = os.path.join(DATA_DIR, file)
        file_ext = os.path.splitext(file)[1].lower()
        raw_text = read_file(
            file_path,
            use_ocr=USE_OCR,
            ocr_lang=OCR_LANG,
            gpu=USE_GPU_OCR,
            dpi_scale=DPI_SCALE,
            min_text_ratio=MIN_TEXT_RATIO,
            ocr_page_limit=OCR_PAGE_LIMIT
        )
        if not raw_text or len(raw_text.strip()) < 50:
            print(f" ⚠️ Пропущен: {file}")
            continue

        file_meta = extract_metadata(file_path)
        text = normalize_text(raw_text)

        candidates = extract_candidate_species(text)
        for c in candidates:
            lemma = lemmatize_phrase(c)
            species_counter[lemma] += 1

        chunks = semantic_chunk(text)
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append("passage: " + chunk)
            metadata.append({
                "source": file,
                "format": file_ext.lstrip("."),
                "title": file_meta.get("title"),
                "author": file_meta.get("author"),
                "length": len(chunk)
            })

        print(f" ✅ {file}: {len(chunks)} чанков, {len(candidates)} видов")

    print(f"\n📊 Итого чанков: {len(all_chunks)}")
    print(f"📊 Уникальных видов: {len(species_counter)}")

    # ВИДЫ
    species_final = [sp for sp, count in species_counter.items() if count >= MIN_SPECIES_OCC]
    species_final.sort()
    with open(SPECIES_PATH, "w", encoding="utf-8") as f:
        json.dump(species_final, f, ensure_ascii=False, indent=2)
    print(f"🌱 Сохранено видов (≥{MIN_SPECIES_OCC} вхождений): {len(species_final)}")

    # CHECKPOINT
    print("💾 Сохраняем чанки и метаданные...")
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("✅ Checkpoint сохранён!")

# ==========================
# EMBEDDINGS + NPY (ПРАВИЛЬНОЕ МЕСТО!)
# ==========================
EMB_NPY_PATH = "all_embeddings.npy"
print("\n🔢 Работа с эмбеддингами...")

if os.path.exists(EMB_NPY_PATH):
    print(f"Найден {EMB_NPY_PATH} → загружаем")
    embeddings = np.load(EMB_NPY_PATH).astype("float32")
    if embeddings.shape[0] != len(all_chunks):
        print("!!! Несоответствие: эмбеддингов ≠ чанков")
        print("Удалите all_embeddings.npy и запустите заново (с --force если нужно)")
        raise ValueError("Несоответствие размерностей")
    print(f"Загружено {embeddings.shape[0]} эмбеддингов из .npy")
else:
    print("Файл .npy не найден → вычисляем эмбеддинги")
    embeddings = embedder.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")
    print(f"Сохраняем эмбеддинги → {EMB_NPY_PATH}")
    np.save(EMB_NPY_PATH, embeddings)
    print(f"Сохранено {embeddings.shape}")

# ==========================
# FAISS INDEX
# ==========================
dimension = embeddings.shape[1]
n_vectors = embeddings.shape[0]
print(f"\n📐 Векторов: {n_vectors}, Размерность: {dimension}")

if n_vectors < 5000:
    index = faiss.IndexFlatIP(dimension)
    print("✅ Индекс: IndexFlatIP (точный поиск)")
else:
    nlist = max(32, min(int(np.sqrt(n_vectors)), 512))
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    print(f"✅ Индекс: IndexIVFFlat (nlist={nlist})")

index.add(embeddings)

if isinstance(index, faiss.IndexIVFFlat):
    index.nprobe = 32
    print("✅ nprobe = 32 (лучшее качество поиска)")

faiss.write_index(index, "plants.index")
print("💾 Индекс сохранён: plants.index")

# ==========================
# ТЕСТ ПОИСКА
# ==========================
print("\n🔍 Тест поиска:")
test_queries = ["роза садовая", "помидоры уход", "деревья плодовые"]
for query in test_queries:
    q_emb = embedder.encode(["query: " + query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, 3)
    print(f"\nЗапрос: {query}")
    for i, (dist, idx) in enumerate(zip(D[0], I[0])):
        if idx < len(all_chunks):
            text = all_chunks[idx].replace("passage: ", "")[:120]
            source = metadata[idx].get("source", "unknown")
            fmt = metadata[idx].get("format", "txt")
            print(f" {i + 1}. {dist:.4f} | [{fmt}] {source}: {text}...")

print("\n✅ INDEX READY 🚀")