#!/usr/bin/env python3
"""
Скрипт автоматической настройки Birthday Bot с поддержкой Hugging Face.
Создает conda окружение, устанавливает зависимости и настраивает конфигурацию.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml

def run_command(command, description):
    """Выполнение команды с проверкой результата."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - выполнено")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ошибка: {e}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return None

def check_system_dependencies():
    """Проверка системных зависимостей."""
    print("\n🔍 Проверка системных зависимостей...")
    
    # Проверка conda
    if not shutil.which("conda"):
        print("❌ Conda не найдена! Установите Miniconda или Anaconda.")
        print("📥 Скачать: https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Проверка git
    if not shutil.which("git"):
        print("❌ Git не найден! Установите Git.")
        return False
    
    # Проверка FFmpeg
    if not shutil.which("ffmpeg"):
        print("⚠️ FFmpeg не найден! Рекомендуется установить для обработки аудио.")
        print("Ubuntu/Debian: sudo apt install ffmpeg")
        print("macOS: brew install ffmpeg")
        print("Windows: choco install ffmpeg")
        print("Можно продолжить без FFmpeg - он будет установлен через conda.")
    
    print("✅ Основные зависимости проверены")
    return True

def create_directories():
    """Создание необходимых директорий."""
    print("\n📁 Создание директорий...")
    
    directories = [
        "temp",
        "temp/audio", 
        "temp/images",
        "logs",
        "docs/examples/voice_examples",
        "docs/examples/text_examples",
        "docs/images/screenshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📂 Создана директория: {directory}")
    
    # Создаем .gitkeep файлы для пустых директорий
    gitkeep_dirs = ["temp/audio", "temp/images", "logs"]
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch()

def setup_conda_environment():
    """Создание и настройка conda окружения."""
    env_name = "amikhalev_hb_2025_06"
    
    print(f"\n🐍 Настройка conda окружения '{env_name}'...")
    
    # Проверяем, существует ли окружение
    result = run_command(f"conda info --envs | grep {env_name}", "Проверка существующего окружения")
    
    if result and env_name in result:
        print(f"⚠️ Окружение '{env_name}' уже существует")
        response = input("Пересоздать окружение? (y/N): ").lower()
        if response == 'y':
            run_command(f"conda env remove -n {env_name}", f"Удаление существующего окружения {env_name}")
        else:
            print("✅ Используем существующее окружение")
            return True
    
    # Создаем окружение из environment.yml
    if Path("environment.yml").exists():
        success = run_command(f"conda env create -f environment.yml", "Создание окружения из environment.yml")
    else:
        # Создаем базовое окружение
        success = run_command(f"conda create -n {env_name} python=3.9 -y", "Создание базового окружения")
        if success:
            # Устанавливаем зависимости pip
            run_command(f"conda run -n {env_name} pip install -r requirements.txt", "Установка Python зависимостей")
    
    return success is not None

def setup_configuration():
    """Настройка конфигурационных файлов."""
    print("\n⚙️ Настройка конфигурации...")
    
    # Создаем .env файл если его нет
    if not Path(".env").exists():
        if Path(".env.example").exists():
            shutil.copy(".env.example", ".env")
            print("📝 Создан файл .env из .env.example")
        else:
            create_basic_env_file()
    
    # Создаем конфигурацию в ~/.ssh/bot.yaml
    config_dir = Path.home() / ".ssh"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "bot.yaml"
    
    if not config_file.exists():
        create_basic_config_file(config_file)
        print(f"📝 Создан файл конфигурации: {config_file}")
        print("🔑 Не забудьте указать токены в конфигурационном файле!")

def create_basic_env_file():
    """Создание базового .env файла."""
    env_content = """# Birthday Bot Configuration
# Заполните своими значениями

# Telegram Bot Token (обязательно)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here

# Hugging Face API Key (обязательно)
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Настройки Whisper
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_LANGUAGE=ru

# Настройки безопасности
MAX_VOICE_DURATION=60
RATE_LIMIT_MESSAGES=10

# Логирование
LOG_LEVEL=INFO
"""
    
    with open(".env", "w", encoding="utf-8") as f:
        f.write(env_content)

def create_basic_config_file(config_file):
    """Создание базового конфигурационного файла."""
    config_content = {
        "bot": {
            "token": "your_telegram_bot_token_here"
        },
        "huggingface": {
            "api_key": "your_huggingface_api_key_here",
            "model": "black-forest-labs/FLUX.1-dev",
            "timeout": 60
        },
        "whisper": {
            "model": "small",
            "device": "cpu", 
            "language": "ru"
        },
        "security": {
            "max_voice_duration": 60,
            "rate_limit_messages": 10
        },
        "logging": {
            "level": "INFO"
        }
    }
    
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config_content, f, default_flow_style=False, allow_unicode=True)

def create_run_scripts():
    """Создание скриптов для запуска."""
    print("\n📜 Создание скриптов запуска...")
    
    # Скрипт для разработки
    dev_script = """#!/bin/bash
# Скрипт для запуска в режиме разработки

echo "🚀 Запуск Birthday Bot в режиме разработки..."

# Активируем conda окружение
source $(conda info --base)/etc/profile.d/conda.sh
conda activate amikhalev_hb_2025_06

# Проверяем конфигурацию
echo "🔍 Проверка конфигурации..."
python src/main.py --validate-only

if [ $? -eq 0 ]; then
    echo "✅ Конфигурация корректна"
    echo "🔄 Запуск с автоперезапуском..."
    python run_bot.py --debug
else
    echo "❌ Ошибка конфигурации! Проверьте настройки."
    exit 1
fi
"""
    
    # Скрипт для продакшена
    prod_script = """#!/bin/bash
# Скрипт для запуска в продакшене

echo "🚀 Запуск Birthday Bot..."

# Активируем conda окружение
source $(conda info --base)/etc/profile.d/conda.sh
conda activate amikhalev_hb_2025_06

# Запускаем бота
python src/main.py
"""
    
    # Создаем скрипты
    with open("start_dev.sh", "w") as f:
        f.write(dev_script)
    
    with open("start_prod.sh", "w") as f:
        f.write(prod_script)
    
    # Делаем скрипты исполняемыми
    os.chmod("start_dev.sh", 0o755)
    os.chmod("start_prod.sh", 0o755)
    
    print("✅ Созданы скрипты запуска: start_dev.sh, start_prod.sh")

def print_next_steps():
    """Вывод инструкций по следующим шагам."""
    print("\n" + "="*60)
    print("🎉 УСТАНОВКА ЗАВЕРШЕНА!")
    print("="*60)
    
    print("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
    print("\n1. 🤖 Создайте Telegram бота:")
    print("   • Найдите @BotFather в Telegram")
    print("   • Отправьте команду /newbot")
    print("   • Следуйте инструкциям")
    print("   • Сохраните полученный токен")
    
    print("\n2. 🔑 Получите Hugging Face API ключ:")
    print("   • Зарегистрируйтесь на https://huggingface.co")
    print("   • Перейдите в https://huggingface.co/settings/tokens")
    print("   • Создайте новый токен")
    
    print("\n3. ⚙️ Настройте конфигурацию:")
    print("   • Отредактируйте ~/.ssh/bot.yaml")
    print("   • Или отредактируйте .env файл")
    print("   • Укажите ваши токены")
    
    print("\n4. 🚀 Запустите бота:")
    print("   • Для разработки: ./start_dev.sh")
    print("   • Для продакшена: ./start_prod.sh")
    print("   • Или вручную: conda activate amikhalev_hb_2025_06 && python run_bot.py")
    
    print("\n5. ✅ Проверьте работу:")
    print("   • Отправьте /start вашему боту")
    print("   • Попробуйте отправить текстовое сообщение")
    print("   • Попробуйте отправить голосовое сообщение")
    
    print("\n📚 ПОЛЕЗНЫЕ КОМАНДЫ:")
    print("   • Проверка конфигурации: python src/main.py --validate-only")
    print("   • Просмотр логов: tail -f logs/*.log")
    print("   • Остановка бота: Ctrl+C")
    
    print("\n🔧 ФАЙЛЫ КОНФИГУРАЦИИ:")
    print("   • ~/.ssh/bot.yaml - основная конфигурация")
    print("   • .env - переменные окружения")
    print("   • environment.yml - conda окружение")
    
    print("\n❓ ПОМОЩЬ:")
    print("   • README.md - подробная документация")
    print("   • docs/ - дополнительная документация")
    print("   • GitHub Issues для сообщений о багах")

def main():
    """Основная функция установки."""
    print("🎉 Birthday Bot - Автоматическая установка")
    print("="*50)
    
    # Проверяем системные зависимости
    if not check_system_dependencies():
        print("\n❌ Установка прервана из-за отсутствующих зависимостей")
        sys.exit(1)
    
    # Создаем директории
    create_directories()
    
    # Настраиваем conda окружение
    if not setup_conda_environment():
        print("\n❌ Ошибка создания conda окружения")
        sys.exit(1)
    
    # Настраиваем конфигурацию
    setup_configuration()
    
    # Создаем скрипты запуска
    create_run_scripts()
    
    # Выводим следующие шаги
    print_next_steps()

if __name__ == "__main__":
    main()
    