"""
Модуль настройки логирования для Birthday Bot.
Обеспечивает централизованное логирование с ротацией файлов.
Модульный подход для современной архитектуры.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# Глобальный реестр логгеров для модульного подхода
_module_loggers: Dict[str, logging.Logger] = {}

def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_format: Optional[str] = None,
    log_dir: str = "logs",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_detailed: bool = None
) -> logging.Logger:
    """
    Настройка логгера с файловым выводом и ротацией.
    
    Args:
        name: Имя логгера
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Формат сообщений лога
        log_dir: Директория для файлов логов
        max_file_size: Максимальный размер файла лога
        backup_count: Количество архивных файлов логов
        console_detailed: Детальный вывод в консоль (с временем). None = авто
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    # Создаем директорию для логов
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Получаем логгер
    logger = logging.getLogger(name)
    
    # ИСПРАВЛЕНИЕ: Если логгер уже настроен, НЕ добавляем обработчики повторно
    if logger.handlers:
        return logger
    
    # Устанавливаем уровень логирования
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(log_level_map.get(log_level.upper(), logging.INFO))
    
    # Отключаем распространение к корневому логгеру
    logger.propagate = False
    
    # Определяем, нужен ли детальный вывод в консоль
    if console_detailed is None:
        # Автоопределение: детальный вывод для DEBUG режима
        console_detailed = (log_level.upper() == "DEBUG" or 
                          os.getenv('LOG_DETAILED', 'false').lower() == 'true')
    
    # Формат сообщений по умолчанию с файлом и номером строки
    if log_format is None:
        log_format = "%(asctime)s [%(name)s](%(levelname)s) %(filename)s:%(lineno)d: %(message)s"
    
    # Создаем форматтеры
    # Подробный формат для файлов
    detailed_formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Формат для консоли (детальный или упрощенный)
    if console_detailed:
        console_formatter = detailed_formatter  # Полный формат
    else:
        # Упрощенный формат для консоли
        console_format = "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s"
        console_formatter = logging.Formatter(
            console_format,
            datefmt="%H:%M:%S"
        )
    
    # Настройка консольного вывода
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Настройка файлового вывода с ротацией (подробный формат)
    log_file = os.path.join(log_dir, f"{name.replace('.', '_')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Настройка отдельного файла для ошибок (подробный формат)
    error_log_file = os.path.join(log_dir, "error.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    return logger

def clear_logger_handlers(logger_name: str = None):
    """
    Очистка обработчиков логгера для предотвращения дублирования.
    
    Args:
        logger_name: Имя логгера (если None, очищает все логгеры проекта)
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    else:
        # Очищаем все логгеры проекта
        for name in list(logging.Logger.manager.loggerDict.keys()):
            if name.startswith("birthday_bot"):
                logger = logging.getLogger(name)
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()

def reset_module_loggers():
    """
    Сброс кэша модульных логгеров.
    Полезно при перезапуске или отладке.
    """
    global _module_loggers
    
    # Закрываем все обработчики
    for logger in _module_loggers.values():
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
    
    # Очищаем кэш
    _module_loggers.clear()

def get_module_logger(module_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Получение логгера для конкретного модуля с предотвращением дублирования.
    
    Args:
        module_name: Имя модуля
        log_level: Уровень логирования
        
    Returns:
        logging.Logger: Настроенный логгер для модуля
    """
    logger_name = f"birthday_bot.{module_name}"
    
    # ИСПРАВЛЕНИЕ: Проверяем кэш и существующие логгеры
    if module_name in _module_loggers:
        return _module_loggers[module_name]
    
    # Проверяем, не существует ли уже логгер с таким именем
    existing_logger = logging.getLogger(logger_name)
    if existing_logger.handlers:
        _module_loggers[module_name] = existing_logger
        return existing_logger
    
    # Создаем новый логгер
    logger = setup_logger(name=logger_name, log_level=log_level)
    _module_loggers[module_name] = logger
    return logger

def get_logger_for_module(module_path: str) -> logging.Logger:
    """
    Получение логгера на основе пути модуля.
    Автоматически определяет имя модуля из __file__.
    
    Args:
        module_path: Путь к файлу модуля (__file__)
        
    Returns:
        logging.Logger: Логгер для модуля
    """
    # Извлекаем имя модуля из пути
    module_file = Path(module_path)
    if module_file.stem == "__init__":
        module_name = module_file.parent.name
    else:
        module_name = module_file.stem
    
    return get_module_logger(module_name)

def setup_project_logging(base_name: str = "birthday_bot", 
                         log_level: str = "INFO") -> logging.Logger:
    """
    Настройка логирования для всего проекта.
    
    Args:
        base_name: Базовое имя для логгеров
        log_level: Уровень логирования
        
    Returns:
        logging.Logger: Корневой логгер проекта
    """
    # Настраиваем корневой логгер проекта
    root_logger = setup_logger(base_name, log_level)
    
    # Устанавливаем уровень для всех дочерних логгеров
    logging.getLogger(base_name).setLevel(getattr(logging, log_level.upper()))
    
    return root_logger

def set_global_log_level(level: str):
    """
    Установка глобального уровня логирования для всех логгеров.
    
    Args:
        level: Уровень логирования
    """
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    numeric_level = log_level_map.get(level.upper(), logging.INFO)
    
    # Устанавливаем уровень для всех существующих логгеров
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
    
    # Устанавливаем уровень для корневого логгера
    logging.getLogger().setLevel(numeric_level)

def log_execution_time(logger_name: str = "birthday_bot"):
    """
    Декоратор для логирования времени выполнения функций.
    
    Args:
        logger_name: Имя логгера для вывода
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_module_logger(logger_name)
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"EXEC_TIME | {func.__name__} | {execution_time:.3f}s")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"EXEC_ERROR | {func.__name__} | {execution_time:.3f}s | {str(e)}")
                raise
                
        return wrapper
    return decorator

# ИСПРАВЛЕНИЕ: Простые функции вместо класса для правильного отображения источника
def get_bot_logger() -> logging.Logger:
    """Получить логгер для бота."""
    return get_module_logger("bot")

def get_speech_logger() -> logging.Logger:
    """Получить логгер для модуля речи."""
    return get_module_logger("speech")

def get_image_logger() -> logging.Logger:
    """Получить логгер для модуля изображений."""
    return get_module_logger("image")

def get_handlers_logger() -> logging.Logger:
    """Получить логгер для обработчиков."""
    return get_module_logger("handlers")

# Функция для совместимости с остальным кодом
def get_logger(module_name: str, log_file: str = None) -> logging.Logger:
    """
    Получение логгера для совместимости с существующим кодом.
    
    Args:
        module_name: Имя модуля
        log_file: Имя файла лога (игнорируется для совместимости)
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    return get_module_logger(module_name)

# Класс для расширенного логирования (только для специальных случаев)
class ModularLogger:
    """Продвинутый логгер с дополнительными методами для бота."""
    
    def __init__(self, module_name: str):
        """
        Инициализация модульного логгера.
        
        Args:
            module_name: Имя модуля
        """
        self.logger = get_module_logger(module_name)
        self.module_name = module_name
    
    def log_user_action(self, user_id: int, username: str, action: str, details: str = ""):
        """
        Логирование действий пользователя.
        
        Args:
            user_id: ID пользователя
            username: Имя пользователя
            action: Действие
            details: Дополнительные детали
        """
        message = f"USER_ACTION | ID:{user_id} | @{username} | {action}"
        if details:
            message += f" | {details}"
        self.logger.info(message)
    
    def log_message_processing(self, user_id: int, message_type: str, processing_time: float):
        """
        Логирование обработки сообщений.
        
        Args:
            user_id: ID пользователя
            message_type: Тип сообщения (text, voice, etc.)
            processing_time: Время обработки в секундах
        """
        self.logger.info(
            f"MESSAGE_PROCESSING | USER:{user_id} | TYPE:{message_type} | "
            f"TIME:{processing_time:.2f}s"
        )
    
    def log_speech_recognition(self, user_id: int, audio_duration: float, 
                              recognition_time: float, success: bool):
        """
        Логирование распознавания речи.
        
        Args:
            user_id: ID пользователя
            audio_duration: Длительность аудио
            recognition_time: Время распознавания
            success: Успешность распознавания
        """
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"SPEECH_RECOGNITION | USER:{user_id} | DURATION:{audio_duration:.1f}s | "
            f"PROCESSING:{recognition_time:.2f}s | STATUS:{status}"
        )
    
    def log_image_generation(self, user_id: int, prompt_length: int, 
                           generation_time: float, success: bool):
        """
        Логирование генерации изображений.
        
        Args:
            user_id: ID пользователя
            prompt_length: Длина промпта
            generation_time: Время генерации
            success: Успешность генерации
        """
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"IMAGE_GENERATION | USER:{user_id} | PROMPT_LEN:{prompt_length} | "
            f"TIME:{generation_time:.2f}s | STATUS:{status}"
        )
    
    def log_error_with_context(self, error: Exception, context: dict):
        """
        Логирование ошибки с контекстом.
        
        Args:
            error: Исключение
            context: Контекстная информация
        """
        context_str = " | ".join([f"{k}:{v}" for k, v in context.items()])
        self.logger.error(f"ERROR | {str(error)} | CONTEXT: {context_str}", exc_info=True)

# Экспорт основных функций
__all__ = [
    'setup_logger', 
    'get_module_logger',
    'get_logger_for_module',
    'setup_project_logging',
    'clear_logger_handlers',
    'reset_module_loggers',
    'get_bot_logger',
    'get_speech_logger', 
    'get_image_logger',
    'get_handlers_logger',
    'get_logger',  # Для совместимости
    'ModularLogger',
    'set_global_log_level',
    'log_execution_time'
]
