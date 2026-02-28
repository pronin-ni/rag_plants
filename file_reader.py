# file_reader.py
"""
Модуль для чтения файлов различных форматов:
- TXT, FB2, EPUB, PDF, DJVu

OCR через EasyOCR (лучшее качество для русского языка)
"""

import os
import re
from typing import Optional, List
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
import zipfile

# Глобальный кэш для EasyOCR (чтобы не инициализировать каждый раз)
_easyocr_reader = None


def get_easyocr_reader(lang: List[str] = ["ru", "en"], gpu: bool = False):
    """
    Получение/создание EasyOCR reader с кэшированием

    Args:
        lang: Языки распознавания
        gpu: Использовать GPU (если доступен)

    Returns:
        EasyOCR Reader объект
    """
    global _easyocr_reader

    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(lang, gpu=gpu, verbose=False)
        print(f"   ✅ EasyOCR инициализирован (языки: {', '.join(lang)}, GPU: {gpu})")

    return _easyocr_reader


# ==========================
# ЧТЕНИЕ TXT
# ==========================

def read_txt(file_path: str, encoding: str = "utf-8") -> str:
    """Чтение обычного текстового файла"""
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


# ==========================
# ЧТЕНИЕ DOCX
# ==========================

def read_docx(file_path: str) -> str:
    """Чтение файла DOCX с извлечением текста"""
    try:
        from docx import Document
        document = Document(file_path)
        full_text = []

        for para in document.paragraphs:
            if para.text.strip():
                full_text.append(para.text)

        return "\n\n".join(full_text) if full_text else ""
    except Exception as e:
        print(f"⚠️ Ошибка чтения DOCX {file_path}: {type(e).__name__}: {e}")
        return ""


# ==========================
# ЧТЕНИЕ FB2
# ==========================

def read_fb2(file_path: str) -> str:
    """Чтение FB2 файла с извлечением текста"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        soup = BeautifulSoup(content, "lxml-xml")

        for tag in soup(["stylesheet", "title-info", "coverpage", "annotation", "epigraph"]):
            tag.decompose()

        texts = []
        for section in soup.find_all(["section", "p", "title", "poem", "stanza"]):
            text = section.get_text(strip=True)
            if text and len(text) > 10:
                texts.append(text)

        if texts:
            return "\n\n".join(texts)

        body = soup.find("body")
        if body:
            return body.get_text(separator="\n\n", strip=True)

        return ""

    except Exception as e:
        print(f"⚠️ Ошибка чтения FB2 {file_path}: {type(e).__name__}: {e}")
        return ""


# ==========================
# ЧТЕНИЕ EPUB
# ==========================

def read_epub(file_path: str) -> str:
    """Чтение EPUB файла с извлечением текста из глав"""
    try:
        book = epub.read_epub(file_path)
        chapters = []

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            name = item.get_name().lower()
            if any(skip in name for skip in ["toc", "nav", "cover", "style", "opfs"]):
                continue

            soup = BeautifulSoup(item.get_content(), "html.parser", from_encoding="utf-8")

            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()

            title_tag = soup.find(["h1", "h2", "h3"])
            if title_tag:
                title = title_tag.get_text(strip=True)
                if title and len(title) < 200:
                    chapters.append(f"# {title}")

            content = soup.find(["article", "main"]) or soup.find("body") or soup
            text = content.get_text(separator="\n", strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)

            if text.strip() and len(text.strip()) > 50:
                chapters.append(text.strip())

        return "\n\n---\n\n".join(chapters) if chapters else ""

    except zipfile.BadZipFile:
        print(f"⚠️ Файл {file_path} не является валидным EPUB")
        return ""
    except Exception as e:
        print(f"⚠️ Ошибка чтения EPUB {file_path}: {type(e).__name__}: {e}")
        return ""


# ==========================
# ЧТЕНИЕ PDF (с EasyOCR)
# ==========================

def read_pdf(file_path: str, use_ocr: bool = True, ocr_lang: List[str] = ["ru", "en"],
             gpu: bool = False, dpi_scale: float = 3.0,
             min_text_ratio: float = 0.1,  # ✅ Новый параметр: мин. доля текста для пропуска OCR
             ocr_page_limit: int = None) -> str:  # ✅ Лимит страниц для OCR
    """
    Чтение PDF файла с умным определением: текстовый слой или скан

    Args:
        file_path: Путь к PDF файлу
        use_ocr: Использовать OCR если текстовый слой отсутствует
        ocr_lang: Языки для OCR
        gpu: Использовать GPU для OCR
        dpi_scale: Масштаб рендера для OCR (3.0 = ~300 DPI)
        min_text_ratio: Мин. доля символов на странице для считания её "текстовой" (0.1 = 10%)
        ocr_page_limit: Макс. количество страниц для OCR (None = без лимита)

    Returns:
        Текст документа
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        pages = []
        ocr_pages_count = 0
        text_pages_count = 0
        total_pages = len(doc)

        print(f"   📄 PDF: {total_pages} страниц, анализ...")

        for page_num in range(total_pages):
            page = doc[page_num]

            # ==========================================
            # УМНОЕ ОПРЕДЕЛЕНИЕ: текст или скан?
            # ==========================================

            # 1. Получаем текст из текстового слоя
            text = page.get_text("text")

            # 2. Считаем "полезные" символы (кириллица, латиница, цифры, базовая пунктуация)
            useful_chars = len(re.findall(r'[А-Яа-яЁёA-Za-z0-9\s\.\,\!\?\;\:\-\(\)]', text))
            total_chars = len(text) if text else 0

            # 3. Определяем, достаточно ли текста чтобы считать страницу "текстовой"
            has_text_layer = False
            if total_chars > 100:  # Минимум 100 символов на странице
                text_ratio = useful_chars / total_chars if total_chars > 0 else 0
                has_text_layer = text_ratio >= min_text_ratio and useful_chars >= 50

            # 4. Дополнительная проверка: есть ли структурированный текст (абзацы, слова)
            if has_text_layer:
                words = text.split()
                # Если есть хотя бы 10 слов средней длины — это точно текст
                meaningful_words = [w for w in words if len(w) >= 3]
                has_text_layer = len(meaningful_words) >= 10

            # ==========================================
            # ОБРАБОТКА СТРАНИЦЫ
            # ==========================================

            if has_text_layer:
                # ✅ Используем текстовый слой
                text_pages_count += 1

            elif use_ocr:
                # 📸 Скан — используем OCR
                # Проверяем лимит страниц для OCR
                if ocr_page_limit and ocr_pages_count >= ocr_page_limit:
                    print(f"   ⚠️ Достигнут лимит OCR ({ocr_page_limit} стр.), пропускаем остальные")
                    # Пробуем хоть что-то взять из текстового слоя
                    if text.strip():
                        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
                        text = re.sub(r'\n{3,}', '\n\n', text)
                        if text.strip() and len(text.strip()) > 50:
                            pages.append(f"--- Страница {page_num + 1} ---\n{text.strip()}")
                    continue

                print(f"   📸 Стр. {page_num + 1}/{total_pages}: скан → OCR...", end="\r")
                ocr_text = ocr_page_easyocr(page, lang=ocr_lang, gpu=gpu, dpi_scale=dpi_scale)
                print(f"   ✅ Стр. {page_num + 1}: OCR завершён" + " " * 20, end="\r")

                if ocr_text.strip():
                    text = ocr_text
                    ocr_pages_count += 1
                else:
                    # Fallback: берём что есть из текстового слоя
                    if text.strip():
                        ocr_pages_count += 1  # Считаем как попытку OCR
            else:
                # OCR отключён, берём что есть
                pass

            # ==========================================
            # ОЧИСТКА И СОХРАНЕНИЕ
            # ==========================================

            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            if text and len(text) > 50:
                pages.append(f"--- Страница {page_num + 1} ---\n{text}")

        # ✅ Закрываем документ ТОЛЬКО ПОСЛЕ обработки всех страниц
        doc.close()

        # 📊 Статистика
        if ocr_pages_count > 0:
            print(f"\n   ✅ EasyOCR: {ocr_pages_count}/{total_pages} страниц")
        if text_pages_count > 0:
            print(f"   ✅ Текстовый слой: {text_pages_count}/{total_pages} страниц")

        return "\n\n".join(pages)

    except ValueError as e:
        if "document closed" in str(e).lower():
            print(f"\n   ⚠️ Ошибка: документ закрыт преждевременно. Попробуйте уменьшить batch_size.")
        else:
            print(f"\n   ⚠️ Ошибка PDF: {e}")
        return ""
    except ImportError:
        print(f"⚠️ PyMuPDF не установлен: pip install pymupdf")
        return ""
    except Exception as e:
        print(f"⚠️ Ошибка чтения PDF {file_path}: {type(e).__name__}: {e}")
        return ""


# ==========================
# ЧТЕНИЕ DJVU (с EasyOCR)
# ==========================

def read_djvu(file_path: str, use_ocr: bool = True, ocr_lang: List[str] = ["ru", "en"],
              gpu: bool = False, dpi_scale: float = 3.0,
              min_text_ratio: float = 0.15,
              ocr_page_limit: int = 50) -> str:
    """
    Чтение DJVU файла через DjVuLibre CLI (Windows-совместимо)

    Стратегия:
    1. djvutxt — извлечение текстового слоя (если есть)
    2. djvupdf + read_pdf() + OCR — если текстового слоя нет

    Args:
        file_path: Путь к DJVU файлу
        use_ocr: Использовать OCR если текстовый слой отсутствует
        ocr_lang: Языки для OCR
        gpu: Использовать GPU для OCR
        dpi_scale: Масштаб рендера для OCR
        min_text_ratio: Мин. доля текста для пропуска OCR (передаётся в read_pdf)
        ocr_page_limit: Лимит страниц для OCR (передаётся в read_pdf)

    Returns:
        Текст документа
    """
    import subprocess
    import shutil
    import tempfile
    import os

    print(f"   📖 Чтение DJVU через DjVuLibre CLI...")

    # ==========================================
    # ПОИСК djvutxt и djvupdf в системе
    # ==========================================

    djvutxt_path = shutil.which("djvutxt")
    djvupdf_path = shutil.which("djvupdf")

    # Если не найдено через which — пробуем стандартные пути Windows
    if not djvutxt_path or not djvupdf_path:
        potential_paths = [
            r"C:\Program Files\DjVuLibre",
            r"C:\Program Files (x86)\DjVuLibre",
            r"C:\DjVuLibre",
        ]
        for base_path in potential_paths:
            if not djvutxt_path:
                candidate = os.path.join(base_path, "djvutxt.exe")
                if os.path.exists(candidate):
                    djvutxt_path = candidate
            if not djvupdf_path:
                candidate = os.path.join(base_path, "djvupdf.exe")
                if os.path.exists(candidate):
                    djvupdf_path = candidate

    # ==========================================
    # ПОПЫТКА 1: djvutxt — извлечение текстового слоя
    # ==========================================

    if djvutxt_path:
        try:
            print(f"   🔍 Попытка извлечь текст через djvutxt...")

            result = subprocess.run(
                [djvutxt_path, file_path],
                capture_output=True,
                text=True,
                timeout=120,  # 2 минуты на файл
                encoding="utf-8",
                errors="ignore",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0  # Без окна на Windows
            )

            if result.returncode == 0:
                text = result.stdout

                # Очистка от служебных символов djvutxt
                text = re.sub(r'\[\d+\]', '', text)  # Убираем [1], [2] — номера страниц
                text = re.sub(r'<<\d+>>', '', text)  # Убираем <<123>> — координаты
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = text.strip()

                # Проверка: достаточно ли текста?
                useful_chars = len(re.findall(r'[А-Яа-яЁёA-Za-z0-9\s\.\,\!\?\;\:\-\(\)]', text))
                total_chars = len(text) if text else 0
                words = [w for w in text.split() if len(w) >= 3]

                if total_chars > 500 and useful_chars / max(total_chars, 1) >= 0.1 and len(words) >= 30:
                    print(f"   ✅ Текстовый слой извлечён: {len(text)} символов")
                    return text

            else:
                print(f"   ⚠️ djvutxt вернул код {result.returncode}")
                if result.stderr:
                    print(f"      stderr: {result.stderr[:200]}")

        except subprocess.TimeoutExpired:
            print(f"   ⚠️ djvutxt таймаут (>120 сек)")
        except FileNotFoundError:
            print(f"   ⚠️ djvutxt не найден")
        except Exception as e:
            print(f"   ⚠️ Ошибка djvutxt: {type(e).__name__}: {e}")

    else:
        print(f"   ⚠️ djvutxt не найден в PATH или стандартных путях")

    # ==========================================
    # ПОПЫТКА 2: djvupdf + read_pdf() + OCR
    # ==========================================

    if djvupdf_path and use_ocr:
        try:
            print(f"   🔄 Текстовый слой не найден → конвертирую в PDF + OCR...")

            # Создаём временный PDF
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                temp_pdf = tmp.name

            # Конвертация DJVU → PDF
            subprocess.run(
                [djvupdf_path, file_path, "-o", temp_pdf],
                check=True,
                capture_output=True,
                timeout=300,  # 5 минут на конвертацию
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            )

            if os.path.exists(temp_pdf):
                print(f"   ✅ Конвертировано в PDF, запускаю OCR...")

                # Чтение PDF с OCR (используем нашу же функцию!)
                text = read_pdf(
                    temp_pdf,
                    use_ocr=True,
                    ocr_lang=ocr_lang,
                    gpu=gpu,
                    dpi_scale=dpi_scale,
                    min_text_ratio=min_text_ratio,
                    ocr_page_limit=ocr_page_limit
                )

                # Удаляем временный файл
                os.unlink(temp_pdf)

                if text and len(text.strip()) > 100:
                    print(f"   ✅ OCR завершён: {len(text)} символов")
                    return text

        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Ошибка конвертации djvupdf: {e}")
            if e.stderr:
                print(f"      stderr: {e.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ Конвертация таймаут (>300 сек)")
        except Exception as e:
            print(f"   ⚠️ Ошибка: {type(e).__name__}: {e}")
        finally:
            # Гарантированная очистка временного файла
            if os.path.exists(temp_pdf):
                try:
                    os.unlink(temp_pdf)
                except:
                    pass

    elif not djvupdf_path:
        print(f"   ⚠️ djvupdf не найден — не могу конвертировать в PDF")

    # ==========================================
    # FALLBACK: Информативное сообщение
    # ==========================================

    print(f"\n   ⚠️ Не удалось извлечь текст из DJVU")
    print(f"   📁 Файл: {os.path.basename(file_path)}")
    print(f"   🔧 Проверьте:")
    print(f"      • DjVuLibre установлен: https://sourceforge.net/projects/djvu/")
    print(f"      • Пути в PATH: djvutxt --version, djvupdf --version")
    print(f"      • Файл не повреждён")

    return ""


# ==========================
# OCR ФУНКЦИИ (EasyOCR)
# ==========================

def ocr_page_easyocr(page, lang: List[str] = ["ru", "en"], gpu: bool = False,
                     dpi_scale: float = 3.0) -> str:
    """
    OCR одной страницы PDF через EasyOCR

    Args:
        page: Страница PyMuPDF
        lang: Языки для распознавания
        gpu: Использовать GPU
        dpi_scale: Масштаб рендера (3.0 = ~300 DPI)

    Returns:
        Распознанный текст
    """
    try:
        import fitz
        import numpy as np

        # Рендерим страницу в изображение
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = page.get_pixmap(matrix=mat)

        # Конвертируем в NumPy array (EasyOCR требует именно это!)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))

        # Получение/создание reader
        reader = get_easyocr_reader(lang=lang, gpu=gpu)

        # Распознавание текста
        results = reader.readtext(img)

        # Сбор текста
        text = "\n".join([r[1] for r in results])

        return text

    except Exception as e:
        print(f"⚠️ EasyOCR ошибка (PDF): {type(e).__name__}: {e}")
        return ""


def ocr_djvu_page_easyocr(page, lang: List[str] = ["ru", "en"], gpu: bool = False,
                          dpi_scale: float = 3.0) -> str:
    """
    OCR одной страницы DJVU через EasyOCR

    Args:
        page: Страница djvu.decode
        lang: Языки для распознавания
        gpu: Использовать GPU
        dpi_scale: Масштаб рендера

    Returns:
        Распознанный текст
    """
    try:
        import numpy as np
        from PIL import Image
        import io

        # Рендерим страницу в изображение
        img = page.render()

        # Конвертируем в NumPy array
        if isinstance(img, Image.Image):
            # PIL Image → NumPy array
            img = np.array(img)
        elif hasattr(img, 'samples'):
            # DjVu pixmap → NumPy array
            img = np.frombuffer(img.samples, dtype=np.uint8).reshape(
                (img.height, img.width, img.n if hasattr(img, 'n') else 3)
            )

        # Получение/создание reader
        reader = get_easyocr_reader(lang=lang, gpu=gpu)

        # Распознавание текста
        results = reader.readtext(img)

        # Сбор текста
        text = "\n".join([r[1] for r in results])

        return text

    except Exception as e:
        print(f"⚠️ EasyOCR ошибка (DJVU): {type(e).__name__}: {e}")
        return ""


# ==========================
# УНИВЕРСАЛЬНОЕ ЧТЕНИЕ
# ==========================

def read_file(file_path: str, encoding: str = "utf-8", use_ocr: bool = True,
              ocr_lang: List[str] = ["ru", "en"], gpu: bool = False,
              dpi_scale: float = 3.0,
              min_text_ratio: float = 0.15,      # ✅ Добавьте
              ocr_page_limit: int = 50) -> Optional[str]:  # ✅ Добавьте
    """
    Универсальная функция чтения файлов с автоопределением формата

    Поддерживаемые расширения:
    - .txt — обычный текст
    - .fb2 — FictionBook 2.0
    - .epub — Electronic Publication
    - .pdf — Portable Document Format (с EasyOCR для сканов)
    - .djvu — DjVu Document (с EasyOCR для сканов)

    Args:
        file_path: Путь к файлу
        encoding: Кодировка для TXT файлов
        use_ocr: Использовать OCR для PDF/DJVU без текстового слоя
        ocr_lang: Языки для OCR (список, например ["ru", "en"])
        gpu: Использовать GPU для OCR
        dpi_scale: Масштаб рендера для OCR (3.0 = ~300 DPI)

    Returns:
        Текст файла или None если не удалось прочитать
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".txt":
            return read_txt(file_path, encoding)
        elif ext == ".fb2":
            return read_fb2(file_path)
        elif ext == ".epub":
            return read_epub(file_path)
        elif ext == ".pdf":
            return read_pdf(file_path, use_ocr=use_ocr, ocr_lang=ocr_lang,
                            gpu=gpu, dpi_scale=dpi_scale, min_text_ratio=min_text_ratio, ocr_page_limit=ocr_page_limit)
        elif ext == ".djvu":
            return read_djvu(file_path, use_ocr=use_ocr, ocr_lang=ocr_lang,
                             gpu=gpu, dpi_scale=dpi_scale, min_text_ratio=min_text_ratio, ocr_page_limit=ocr_page_limit)
        elif ext == ".docx":
            return read_docx(file_path)
        else:
            print(f"⚠️ Неизвестный формат: {ext}")
            return None

    except UnicodeDecodeError:
        if ext == ".txt":
            for enc in ["cp1251", "latin-1", "utf-8-sig"]:
                try:
                    return read_txt(file_path, encoding=enc)
                except:
                    continue
        print(f"⚠️ Не удалось декодировать файл: {file_path}")
        return None

    except Exception as e:
        print(f"⚠️ Критическая ошибка чтения {file_path}: {type(e).__name__}: {e}")
        return None


def get_supported_files(directory: str) -> List[str]:
    """Получение списка всех поддерживаемых файлов в директории"""
    supported_extensions = {".txt", ".fb2", ".epub", ".pdf", ".djvu"}
    files = []

    for f in os.listdir(directory):
        ext = os.path.splitext(f)[1].lower()
        if ext in supported_extensions:
            files.append(f)

    return sorted(files)


def get_file_stats(file_path: str) -> dict:
    """Получение статистики файла"""
    ext = os.path.splitext(file_path)[1].lower()
    size = os.path.getsize(file_path)

    text = read_file(file_path)
    char_count = len(text) if text else 0
    word_count = len(text.split()) if text else 0

    return {
        "path": file_path,
        "format": ext.lstrip("."),
        "size_bytes": size,
        "size_mb": round(size / 1024 / 1024, 2),
        "characters": char_count,
        "words": word_count
    }


def extract_metadata(file_path: str) -> dict:
    """Извлечение метаданных из файла"""
    meta = {
        "title": os.path.splitext(os.path.basename(file_path))[0],
        "author": None,
        "language": "ru",
        "format": os.path.splitext(file_path)[1].lower().lstrip(".")
    }

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".fb2":
            with open(file_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "lxml-xml")

            author_tag = soup.find("author")
            if author_tag:
                first_name = author_tag.find("first-name")
                last_name = author_tag.find("last-name")
                if first_name and last_name:
                    meta["author"] = f"{first_name.text} {last_name.text}"
                elif last_name:
                    meta["author"] = last_name.text

            title_tag = soup.find("book-title")
            if title_tag:
                meta["title"] = title_tag.text.strip()

            lang_tag = soup.find("lang")
            if lang_tag:
                meta["language"] = lang_tag.text.strip()[:2]

        elif ext == ".epub":
            book = epub.read_epub(file_path)

            title = book.get_metadata("DC", "title")
            if title:
                meta["title"] = title[0][0]

            creator = book.get_metadata("DC", "creator")
            if creator:
                meta["author"] = creator[0][0]

            language = book.get_metadata("DC", "language")
            if language:
                meta["language"] = language[0][0][:2]

    except Exception as e:
        print(f"⚠️ Не удалось извлечь метаданные из {file_path}: {e}")

    return meta