# check_djvu.py — улучшенная версия
import subprocess
import shutil
import os

print("🔍 Детальная проверка DjVuLibre\n")

# Пути для поиска
install_paths = [
    r"C:\Program Files\DjVuLibre",
    r"C:\Program Files (x86)\DjVuLibre",
    r"C:\DjVuLibre",
]

tools = {
    "djvutxt": "Извлечение текстового слоя",
    "djvupdf": "Конвертация в PDF (для OCR)",
    "djvuinfo": "Информация о файле",
}

results = {}

for tool, desc in tools.items():
    print(f"🔧 {tool}.exe — {desc}")

    # Поиск
    found_path = None
    path = shutil.which(f"{tool}.exe") or shutil.which(tool)
    if path:
        found_path = path
    else:
        for base in install_paths:
            candidate = os.path.join(base, f"{tool}.exe")
            if os.path.exists(candidate):
                found_path = candidate
                break

    if found_path:
        results[tool] = found_path
        print(f"   ✅ Найден: {found_path}")

        # Проверка запуска
        try:
            result = subprocess.run(
                [found_path, "--version"] if tool != "djvupdf" else [found_path, "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            output = (result.stdout or result.stderr).strip().split('\n')[0]
            print(f"   📋 Ответ: {output[:100]}")
        except Exception as e:
            print(f"   ⚠️ Запуск: {e}")
    else:
        print(f"   ❌ Не найден")

        # Проверка: есть ли файл в папках?
        for base in install_paths:
            candidate = os.path.join(base, f"{tool}.exe")
            if os.path.exists(candidate):
                print(f"   💡 Файл есть в {candidate}, но не в PATH")
                print(f"      Добавьте в PATH или используйте абсолютный путь в коде")
                break
        else:
            print(f"   💡 Файл отсутствует — переустановите DjVuLibre")

    print()

def cal
# Итог
print("=" * 60)
if results.get("djvutxt"):
    print("✅ djvutxt работает — текстовый слой будет извлечён!")
else:
    print("❌ djvutxt не работает — текст из DJVU извлечь не получится")

if results.get("djvupdf"):
    print("✅ djvupdf работает — OCR через конвертацию доступен")
else:
    print("⚠️ djvupdf не найден — OCR fallback будет использован при необходимости")

if len(results) >= 2:
    print("\n🎉 DjVuLibre готов к полноценной работе!")