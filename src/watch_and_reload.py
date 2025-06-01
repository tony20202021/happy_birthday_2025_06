#!/usr/bin/env python
"""
Скрипт запуска Telegram бота с автоматическим перезапуском при изменениях.
Использует watchdog для мониторинга файлов и автоматического перезапуска.
"""

import time
import subprocess
import os
import sys
import signal
import argparse
from pathlib import Path
from typing import List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

# Добавляем путь для импорта модулей проекта (поднимаемся на уровень выше из src)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Настройка логирования для Birthday Bot
from src.utils.logger import get_bot_logger
logger = get_bot_logger()

class BotProcess:
    """Класс для управления процессом бота."""
    
    def __init__(self, script_path: str, process_name: str):
        """
        Инициализация процесса бота.
        
        Args:
            script_path: Путь к скрипту бота
            process_name: Имя процесса для идентификации
        """
        self.script_path = script_path
        self.process_name = process_name
        self.process = None
        self.running = False
    
    def start(self) -> None:
        """Запуск процесса бота."""
        if self.running:
            logger.info("Процесс уже запущен. Пропускаем запуск.")
            return
            
        try:
            logger.info(f"{'='*50}")
            logger.info(f"Запуск процесса бота: {self.script_path}")
            self.process = subprocess.Popen(
                ["python", self.script_path, f"--process-name={self.process_name}"],
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            self.running = True
            logger.info(f"Процесс бота запущен с PID: {self.process.pid}")
        except Exception as e:
            logger.error(f"Ошибка запуска процесса бота: {e}")
    
    def stop(self) -> None:
        """Остановка процесса бота."""
        if not self.running or not self.process:
            logger.info("Нет запущенного процесса для остановки.")
            return
            
        try:
            logger.info(f"Остановка процесса бота с PID: {self.process.pid}")
            
            # Отправляем SIGTERM для корректного завершения
            self.process.send_signal(signal.SIGTERM)
            
            # Даем процессу время на корректное завершение
            for _ in range(5):
                if self.process.poll() is not None:
                    break
                time.sleep(1)
            
            # Если процесс все еще работает, принудительно завершаем
            if self.process.poll() is None:
                logger.warning("Процесс не завершился корректно. Отправляем SIGKILL.")
                self.process.kill()
                self.process.wait()
            
            self.running = False
            logger.info("Процесс бота успешно остановлен.")
        except Exception as e:
            logger.error(f"Ошибка остановки процесса бота: {e}")
            self.running = False
    
    def restart(self) -> None:
        """Перезапуск процесса бота."""
        logger.info(f"{'='*50}")
        logger.info("Перезапуск процесса бота...")
        self.stop()
        time.sleep(2)  # Пауза для завершения всех подпроцессов
        self.start()


class ChangeHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы."""
    
    def __init__(self, bot_process: BotProcess, watch_extensions: List[str], ignore_dirs: List[str]):
        """
        Инициализация обработчика изменений.
        
        Args:
            bot_process: Процесс бота для управления
            watch_extensions: Список расширений файлов для отслеживания
            ignore_dirs: Список директорий для игнорирования
        """
        self.bot_process = bot_process
        self.watch_extensions = watch_extensions
        self.ignore_dirs = ignore_dirs
        self.last_modified_time = time.time()
        self.throttle_interval = 5  # Минимальный интервал между перезапусками
    
    def should_process_event(self, event) -> bool:
        """
        Проверка, должно ли событие быть обработано.
        
        Args:
            event: Событие файловой системы
        
        Returns:
            True если событие должно быть обработано
        """
        current_time = time.time()
        if current_time - self.last_modified_time < self.throttle_interval:
            return False
            
        if not isinstance(event, (FileModifiedEvent, FileCreatedEvent)):
            return False
            
        if not any(event.src_path.endswith(ext) for ext in self.watch_extensions):
            return False
            
        if any(ignore_dir in event.src_path for ignore_dir in self.ignore_dirs):
            return False
            
        return True
    
    def on_modified(self, event):
        """Обработка события модификации файла."""
        if self.should_process_event(event):
            logger.info(f"Обнаружено изменение в файле: {event.src_path}")
            self.last_modified_time = time.time()
            self.bot_process.restart()
    
    def on_created(self, event):
        """Обработка события создания файла."""
        if self.should_process_event(event):
            logger.info(f"Обнаружен новый файл: {event.src_path}")
            self.last_modified_time = time.time()
            self.bot_process.restart()


def cleanup_and_exit():
    """Обработчик сигналов для корректного завершения."""
    logger.info("Получен сигнал завершения. Останавливаем процессы...")
    # Останавливаем процессы Python
    os.system("pkill -f 'python src/main.py'")
    sys.exit(0)


def run_watcher():
    """Запуск файлового наблюдателя."""
    # Настройки по умолчанию
    script_path = "src/main.py"
    watch_paths = ["./", "src", "docs"]
    process_name = "birthday_bot_autoreload"
    watch_extensions = ['.py', '.yaml', '.yml', '.json']
    ignore_dirs = ['__pycache__', '.git', 'env', 'venv', '.env', '.venv', 'logs']
    
    logger.info(f"Запуск наблюдателя для скрипта: {script_path}")
    logger.info(f"Отслеживаемые директории: {watch_paths}")
    logger.info(f"Отслеживаемые расширения: {watch_extensions}")
    logger.info(f"Игнорируемые директории: {ignore_dirs}")
    
    # Инициализация менеджера процесса бота
    bot_process = BotProcess(script_path, process_name)
    
    # Инициализация обработчика событий
    event_handler = ChangeHandler(bot_process, watch_extensions, ignore_dirs)
    
    # Инициализация наблюдателя
    observer = Observer()
    
    # Планирование отслеживания директорий
    for path in watch_paths:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_dir():
            observer.schedule(event_handler, path, recursive=True)
            logger.info(f"Отслеживается директория: {path}")
        else:
            logger.warning(f"Директория не найдена: {path}")
    
    # Запуск наблюдателя
    observer.start()
    
    # Первоначальный запуск бота
    bot_process.start()
    
    # Установка обработчиков сигналов
    signal.signal(signal.SIGINT, lambda s, f: cleanup_and_exit())
    signal.signal(signal.SIGTERM, lambda s, f: cleanup_and_exit())
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Получено прерывание с клавиатуры. Останавливаем...")
        bot_process.stop()
        observer.stop()
    
    observer.join()
    logger.info("Наблюдатель остановлен.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Запуск бота с автоматическим перезапуском')
    parser.add_argument('--debug', action='store_true', help='Включить отладочный режим')
    
    args = parser.parse_args()
    
    if args.debug:
        from src.utils.logger import set_global_log_level
        set_global_log_level("DEBUG")
        logger.info("🔍 Режим отладки включен")
    
    logger.info("🚀 Запуск Birthday Bot с автоматическим перезапуском...")
    run_watcher()
