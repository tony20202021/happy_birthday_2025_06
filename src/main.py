#!/usr/bin/env python
"""
Основной файл Telegram бота для генерации поздравительных картинок.
Точка входа в приложение.
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Добавляем родительскую директорию в sys.path для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))

# Импорты aiogram 3.x
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv

# Импорты модулей проекта
from src.bot.bot_instance import BotManager
from src.bot.handlers import register_handlers
from src.utils.config import config
from src.utils.logger import setup_project_logging, get_module_logger

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация логгера (будет настроен после парсинга аргументов)
logger = None

def validate_environment() -> bool:
    """
    Валидация окружения перед запуском.
    
    Returns:
        bool: True если окружение валидно
    """
    if logger:
        logger.info("🔍 Проверка окружения...")
    
    # Используем встроенную валидацию конфигурации
    config_errors = config.validate()
    
    if config_errors:
        if logger:
            logger.error("❌ Проверка окружения провалена:")
            for error in config_errors:
                logger.error(f"  - {error}")
        else:
            print("❌ Проверка окружения провалена:")
            for error in config_errors:
                print(f"  - {error}")
        return False
    
    # Дополнительные проверки зависимостей
    missing_deps = []
    
    try:
        import whisper
        if logger:
            logger.info("✅ OpenAI Whisper доступен")
    except ImportError:
        missing_deps.append("OpenAI Whisper (pip install openai-whisper)")
    
    try:
        import PIL
        if logger:
            logger.info("✅ Pillow доступен")
    except ImportError:
        missing_deps.append("Pillow (pip install Pillow)")
    
    try:
        import torch
        if logger:
            logger.info("✅ PyTorch доступен")
    except ImportError:
        missing_deps.append("PyTorch (pip install torch)")
    
    try:
        import diffusers
        if logger:
            logger.info("✅ Diffusers доступен")
    except ImportError:
        missing_deps.append("Diffusers (pip install diffusers)")
    
    # Проверяем FFmpeg
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if logger:
            logger.info(f"✅ FFmpeg доступен: {ffmpeg_path}")
    except ImportError:
        missing_deps.append("imageio-ffmpeg (pip install imageio-ffmpeg)")
    except Exception as e:
        missing_deps.append(f"FFmpeg: {e}")
    
    if missing_deps:
        if logger:
            logger.error("❌ Отсутствуют зависимости:")
            for dep in missing_deps:
                logger.error(f"  - {dep}")
        else:
            print("❌ Отсутствуют зависимости:")
            for dep in missing_deps:
                print(f"  - {dep}")
        return False
    
    if logger:
        logger.info("✅ Проверка окружения прошла успешно")
    return True

async def on_startup(dispatcher: Dispatcher, bot: Bot) -> None:
    """
    Действия при запуске бота.
    
    Args:
        dispatcher: Диспетчер aiogram
        bot: Экземпляр бота
    """
    logger.info("=" * 50)
    logger.info("🚀 Запуск Birthday Bot...")
    logger.info("=" * 50)
    
    # Получаем информацию о боте
    try:
        me = await bot.get_me()
        logger.info(f"🤖 Информация о боте:")
        logger.info(f"   ID: {me.id}")
        logger.info(f"   Имя: {me.first_name}")
        logger.info(f"   Username: @{me.username}")
        logger.info(f"   Может работать в группах: {me.can_join_groups}")
    except Exception as e:
        logger.error(f"❌ Ошибка получения информации о боте: {e}")
    
    # Настройка команд бота
    bot_manager = dispatcher.get("bot_manager")
    if bot_manager:
        try:
            await bot_manager.setup_commands()
            logger.info("✅ Команды бота настроены")
        except Exception as e:
            logger.error(f"❌ Ошибка настройки команд бота: {e}")
    else:
        logger.warning("⚠️ Менеджер бота не найден")
    
    # Регистрация обработчиков
    try:
        register_handlers(dispatcher)
        logger.info("✅ Обработчики зарегистрированы")
    except Exception as e:
        logger.error(f"❌ Ошибка регистрации обработчиков: {e}")
        raise
    
    # Логируем статус конфигурации
    status = config.get_status()
    logger.info("📊 Статус системы:")
    for key, value in status.items():
        status_icon = "✅" if value else "❌"
        logger.info(f"   {status_icon} {key}: {value}")
    
    # Очистка при старте если включена
    if config.production.cleanup_on_start:
        try:
            temp_dirs = [
                Path(config.paths.temp_audio),
                Path(config.paths.temp_images)
            ]
            
            cleaned_count = 0
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for temp_file in temp_dir.rglob("*"):
                        if temp_file.is_file():
                            try:
                                temp_file.unlink()
                                cleaned_count += 1
                            except Exception:
                                pass
            
            if cleaned_count > 0:
                logger.info(f"🧹 Очищено {cleaned_count} старых временных файлов")
        except Exception as e:
            logger.warning(f"⚠️ Ошибка очистки при старте: {e}")
    
    logger.info("=" * 50)
    logger.info("🎉 Бот успешно запущен и готов к работе!")
    logger.info("=" * 50)

async def on_shutdown(dispatcher: Dispatcher) -> None:
    """
    Действия при остановке бота.
    
    Args:
        dispatcher: Диспетчер aiogram
    """
    logger.info("=" * 30)
    logger.info("🛑 Остановка бота...")
    logger.info("=" * 30)
    
    # Очистка временных файлов
    try:
        temp_dirs = [
            Path(config.paths.temp_audio),
            Path(config.paths.temp_images)
        ]
        
        cleaned_count = 0
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for temp_file in temp_dir.rglob("*"):
                    if temp_file.is_file():
                        try:
                            temp_file.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.debug(f"Не удалось удалить файл {temp_file}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"✅ Очищено {cleaned_count} временных файлов")
        else:
            logger.info("✅ Временные файлы уже очищены")
            
    except Exception as e:
        logger.error(f"❌ Ошибка очистки временных файлов: {e}")
    
    # Очистка ресурсов бота
    try:
        bot_manager = dispatcher.get("bot_manager")
        if bot_manager:
            await bot_manager.cleanup()
    except Exception as e:
        logger.error(f"❌ Ошибка очистки ресурсов бота: {e}")
    
    logger.info("🏁 Бот успешно остановлен!")

async def main() -> None:
    """
    Основная функция инициализации и запуска бота.
    """
    try:
        # Валидация окружения (использует config.validate())
        if not validate_environment():
            logger.error("❌ Валидация окружения провалена!")
            sys.exit(1)
        
        # Получаем токен из конфигурации
        if not config.bot.token or config.bot.token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            logger.error("❌ Токен бота не настроен!")
            logger.error("Создайте файл ../../secrets.yaml с корректным токеном")
            sys.exit(1)
        
        logger.info("✅ Токен бота загружен из конфигурации")
        
        # Создание экземпляров бота и диспетчера
        bot = Bot(token=config.bot.token)
        storage = MemoryStorage()
        dp = Dispatcher(storage=storage)
        
        logger.info("✅ Экземпляры бота и диспетчера созданы")
        
        # Создание менеджера бота
        bot_manager = BotManager(bot, dp)
        
        # Сохранение объектов в диспетчере
        dp["bot_manager"] = bot_manager
        dp["bot"] = bot
        dp["config"] = config
        
        # Регистрация обработчиков запуска и остановки
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        
        logger.info("✅ Обработчики запуска и остановки зарегистрированы")
        logger.info("🔄 Начало опроса сообщений...")
        
        # Запуск опроса
        await dp.start_polling(
            bot,
            skip_updates=True,  # Пропускаем обновления, накопившиеся за время остановки
        )

    except KeyboardInterrupt:
        logger.info("🔴 Получено прерывание с клавиатуры")
        raise
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при запуске бота: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Birthday Bot - Telegram бот для создания поздравительных картинок',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Примеры использования:
  python src/main.py                    # Обычный запуск
  python src/main.py --debug            # Отладочный режим  
  python src/main.py --validate-only    # Только проверка конфигурации
            """
        )
        
        # Аргументы командной строки
        parser.add_argument('--process-name', type=str, help='Имя процесса для идентификации')
        parser.add_argument('--debug', action='store_true', help='Включить отладочный режим')
        parser.add_argument('--validate-only', action='store_true', help='Только проверить конфигурацию')
        parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                          help='Уровень логирования')
        
        args, unknown = parser.parse_known_args()
        
        # Предупреждение о неизвестных аргументах
        if unknown:
            print(f"⚠️ Неизвестные аргументы: {unknown}")
        
        # Определение уровня логирования
        log_level = config.logging.level  # Берем из конфигурации
        if args.debug:
            log_level = "DEBUG"
        elif args.log_level:
            log_level = args.log_level
        
        # Настройка логирования проекта
        setup_project_logging("birthday_bot", log_level)
        logger = get_module_logger("main", log_level)
        
        # Логирование информации о запуске
        logger.info("🌟 ============ BIRTHDAY BOT STARTUP ============ 🌟")
        
        if args.process_name:
            logger.info(f"🏷️ Идентификатор процесса: {args.process_name}")
        
        pid = os.getpid()
        logger.info(f"🆔 ID процесса (PID): {pid}")
        logger.info(f"📁 Рабочая директория: {os.getcwd()}")
        logger.info(f"🐍 Версия Python: {sys.version}")
        logger.info(f"⏰ Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Уровень логирования: {log_level}")
        logger.info(f"🔧 Режим работы: {'Development' if config.is_development_mode() else 'Production'}")
        
        if args.debug:
            logger.info("🔍 Режим отладки включен")
        
        # Режим только валидации
        if args.validate_only:
            logger.info("🔍 Режим только валидации включен")
            if validate_environment():
                logger.info("✅ Конфигурация валидна - бот готов к запуску")
                
                # Показываем статус конфигурации
                status = config.get_status()
                logger.info("📊 Статус конфигурации:")
                for key, value in status.items():
                    status_icon = "✅" if value else "❌"
                    logger.info(f"   {status_icon} {key}: {value}")
                
                sys.exit(0)
            else:
                logger.error("❌ Валидация конфигурации провалена")
                sys.exit(1)

        # Запуск основного цикла
        asyncio.run(main())
        
    except KeyboardInterrupt:
        if logger:
            logger.info("⏹️ Получен сигнал прерывания")
            logger.info("🔄 Корректное завершение...")
        else:
            print("⏹️ Получен сигнал прерывания")
    except Exception as e:
        if logger:
            logger.error(f"💥 Критическая ошибка в main: {e}", exc_info=True)
        else:
            print(f"💥 Критическая ошибка в main: {e}")
    finally:
        if logger:
            logger.info("🏁 Процесс завершен!")
        else:
            print("🏁 Процесс завершен!")
            