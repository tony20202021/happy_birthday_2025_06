"""
Модуль распознавания речи для Birthday Bot.
Использует OpenAI Whisper для конвертации голосовых сообщений в текст.
Поддержка multi-GPU для параллельного распознавания речи.
"""

import os
import asyncio
import time
import whisper
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from src.utils.config import config
from src.utils.logger import get_speech_logger
from src.speech.audio_processor import AudioProcessor

# Инициализация логгера
logger = get_speech_logger()

class WhisperPool:
    """Пул Whisper моделей для параллельного распознавания речи."""
    
    def __init__(self, gpu_devices: List[str], model_name: str, language: str):
        """
        Инициализация пула Whisper моделей.
        
        Args:
            gpu_devices: Список GPU устройств
            model_name: Название модели Whisper
            language: Язык распознавания
        """
        self.gpu_devices = gpu_devices
        self.model_name = model_name
        self.language = language
        self.models: Dict[str, Any] = {}
        self.available_devices = asyncio.Queue(maxsize=len(gpu_devices))
        self._initialized = False
        
        logger.info(f"🎤 Инициализирован Whisper пул с {len(gpu_devices)} устройствами: {gpu_devices}")
        logger.info(f"   Модель: {model_name}, Язык: {language}")
    
    async def initialize(self):
        """Инициализация всех Whisper моделей для GPU."""
        if self._initialized:
            return
        
        logger.info("📥 Загрузка Whisper моделей на все устройства...")
        
        for device in self.gpu_devices:
            try:
                model = await self._load_model_for_device(device)
                if model:
                    self.models[device] = model
                    await self.available_devices.put(device)
                    logger.info(f"✅ Whisper модель загружена для {device}")
                else:
                    logger.error(f"❌ Не удалось загрузить Whisper модель для {device}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки Whisper модели для {device}: {e}")
        
        self._initialized = True
        logger.info(f"🚀 Whisper пул инициализирован с {len(self.models)} активными устройствами")
    
    async def _load_model_for_device(self, device: str):
        """Загрузка Whisper модели для конкретного устройства."""
        try:
            logger.debug(f"📥 Загрузка Whisper {self.model_name} на {device}")
            
            # Загружаем модель в отдельном потоке
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None,
                whisper.load_model,
                self.model_name,
                device
            )
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки Whisper модели для {device}: {e}")
            return None
    
    @asynccontextmanager
    async def acquire_device(self):
        """Контекстный менеджер для получения Whisper устройства из пула."""
        if not self._initialized:
            await self.initialize()
        
        # Ждем свободное устройство
        device = await self.available_devices.get()
        model = self.models.get(device)
        
        if not model:
            await self.available_devices.put(device)
            raise RuntimeError(f"Whisper модель для {device} недоступна")
        
        try:
            logger.debug(f"🔒 Получен доступ к Whisper {device}")
            yield device, model
        finally:
            # Очищаем память и возвращаем устройство в пул
            self._cleanup_device_memory(device)
            await self.available_devices.put(device)
            logger.debug(f"🔓 Освобожден Whisper {device}")
    
    def _cleanup_device_memory(self, device: str):
        """Очистка памяти конкретного устройства."""
        try:
            import torch
            import gc
            
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            elif device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Ошибка очистки памяти Whisper {device}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса пула Whisper."""
        return {
            "total_devices": len(self.gpu_devices),
            "available_devices": self.available_devices.qsize(),
            "busy_devices": len(self.gpu_devices) - self.available_devices.qsize(),
            "model_name": self.model_name,
            "language": self.language,
            "initialized": self._initialized
        }

# Глобальный пул Whisper (синглтон)
_whisper_pool: Optional[WhisperPool] = None

def get_whisper_pool() -> WhisperPool:
    """Получение глобального пула Whisper."""
    global _whisper_pool
    if _whisper_pool is None:
        gpu_devices = config.speech.gpu_devices
        if not gpu_devices:
            gpu_devices = ["cpu"]  # Fallback
        _whisper_pool = WhisperPool(
            gpu_devices=gpu_devices,
            model_name=config.speech.model_name,
            language=config.speech.language
        )
    return _whisper_pool

class SpeechToText:
    """Класс для распознавания речи с использованием Whisper pool."""
    
    def __init__(self):
        """Инициализация модуля распознавания речи."""
        self.logger = get_speech_logger()
        self.audio_processor = AudioProcessor()
        self.whisper_pool = get_whisper_pool()
        
        # Используем правильные поля из конфигурации
        self.model_name = config.speech.model_name
        self.language = config.speech.language
        
        self.logger.info(f"🎤 Инициализация SpeechToText с multi-GPU поддержкой")
        self.logger.info(f"   Модель: {self.model_name}")
        self.logger.info(f"   Язык: {self.language}")
        self.logger.info(f"   GPU устройства: {config.speech.gpu_devices}")
        
        # Проверяем доступность Whisper
        if not self._check_whisper_availability():
            self.logger.error("❌ OpenAI Whisper недоступен")
            raise ImportError("OpenAI Whisper не установлен. Установите: pip install openai-whisper")
    
    def _check_whisper_availability(self) -> bool:
        """Проверка доступности Whisper."""
        try:
            import whisper
            available_models = whisper.available_models()
            
            if self.model_name not in available_models:
                self.logger.error(f"❌ Модель {self.model_name} недоступна. Доступные: {available_models}")
                return False
            
            self.logger.info(f"✅ Whisper доступен, модель {self.model_name} поддерживается")
            return True
        except ImportError:
            return False
    
    async def transcribe_audio(self, audio_path: str, user_id: int = None) -> Optional[str]:
        """
        Транскрибирует аудиофайл в текст с использованием Whisper pool.
        
        Args:
            audio_path: Путь к аудиофайлу
            user_id: ID пользователя (для логирования)
            
        Returns:
            Optional[str]: Распознанный текст или None при ошибке
        """
        start_time = time.time()
        
        try:
            if not Path(audio_path).exists():
                self.logger.error(f"❌ Аудиофайл не найден: {audio_path}")
                return None
            
            # Валидируем аудиофайл
            if not await self.audio_processor.validate_audio_file(audio_path):
                self.logger.error(f"❌ Аудиофайл не прошел валидацию: {audio_path}")
                return None
            
            # Получаем информацию об аудио
            audio_info = await self.audio_processor.get_audio_info(audio_path)
            duration = audio_info.get("duration_seconds", 0) if audio_info else 0
            
            self.logger.info(f"🎤 Начало транскрибации: {audio_path}")
            self.logger.info(f"   Длительность: {duration:.1f}с")
            if audio_info:
                self.logger.debug(f"   Кодек: {audio_info.get('codec', 'unknown')}")
                self.logger.debug(f"   Частота: {audio_info.get('sample_rate', 'unknown')} Hz")
                self.logger.debug(f"   Каналы: {audio_info.get('channels', 'unknown')}")
            
            # Предобработка аудио для оптимальной работы с Whisper
            processed_audio_path = await self.audio_processor.prepare_for_whisper(audio_path)
            
            if not processed_audio_path:
                self.logger.error("❌ Ошибка предобработки аудио")
                return None
            
            # Инициализируем пул если нужно
            if not self.whisper_pool._initialized:
                await self.whisper_pool.initialize()
            
            # Получаем устройство из пула и транскрибируем
            async with self.whisper_pool.acquire_device() as (device, model):
                self.logger.info(f"🎮 Транскрибация на {device}")
                
                # Запускаем транскрибацию в отдельном потоке
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    self._transcribe_sync, 
                    model, 
                    processed_audio_path
                )
            
            # Удаляем временный обработанный файл если он отличается от оригинала
            if processed_audio_path != audio_path:
                try:
                    Path(processed_audio_path).unlink()
                    self.logger.debug(f"Удален временный файл: {processed_audio_path}")
                except Exception as e:
                    self.logger.warning(f"Не удалось удалить временный файл {processed_audio_path}: {e}")
            
            processing_time = time.time() - start_time
            
            if result:
                self.logger.info(f"✅ Транскрибация завершена за {processing_time:.2f}с")
                self.logger.debug(f"   Распознанный текст: {result[:100]}{'...' if len(result) > 100 else ''}")
                
                # Логируем успешное распознавание
                if user_id:
                    self.logger.info(
                        f"user_id={user_id},"
                        f"audio_duration={duration},"
                        f"recognition_time={processing_time},"
                        f"success=True"
                    )
                
                return result
            else:
                self.logger.warning(f"⚠️ Транскрибация не дала результата за {processing_time:.2f}с")
                
                # Логируем неуспешное распознавание
                if user_id:
                    self.logger.info(
                        f"user_id={user_id},"
                        f"audio_duration={duration},"
                        f"recognition_time={processing_time},"
                        f"success=False"
                    )
                
                return None
                
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(
                f"error={e},"
                f"context={{"
                f'"method": "transcribe_audio",'
                f'"audio_path": "{audio_path}",'
                f'"user_id": {user_id},'
                f'"processing_time": {processing_time}'
                f"}}"
            )
            return None
    
    def _transcribe_sync(self, model, audio_path: str) -> Optional[str]:
        """
        Синхронная транскрибация аудио (выполняется в отдельном потоке).
        
        Args:
            model: Модель Whisper
            audio_path: Путь к аудиофайлу
            
        Returns:
            Optional[str]: Распознанный текст
        """
        try:
            self.logger.debug(f"🔄 Выполнение синхронной транскрибации: {audio_path}")
            
            result = model.transcribe(
                audio_path,
                language=self.language,
                task="transcribe",
                # Дополнительные параметры для улучшения качества
                temperature=0.0,  # Детерминированный результат
                best_of=5,        # Выбор лучшего из 5 попыток
                beam_size=5,      # Размер луча для поиска
                patience=1.0,     # Терпение при поиске
                # Фильтрация коротких сегментов и тишины
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            
            # Извлекаем и очищаем текст
            text = result.get("text", "").strip()
            
            if not text:
                self.logger.debug("❌ Результат транскрибации пуст")
                return None
            
            # Базовая постобработка текста
            text = self._post_process_text(text)
            
            return text if text else None
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в синхронной транскрибации: {e}")
            return None
    
    def _post_process_text(self, text: str) -> str:
        """
        Постобработка распознанного текста.
        
        Args:
            text: Исходный текст
            
        Returns:
            str: Обработанный текст
        """
        if not text:
            return ""
        
        # Удаляем лишние пробелы
        text = " ".join(text.split())
        
        # Удаляем артефакты распознавания
        artifacts = [
            "[BLANK_AUDIO]",
            "[NO_SPEECH]",
            "[MUSIC]",
            "[NOISE]",
            "♪",
            "♫",
            "(music)",
            "(noise)",
            "(silence)"
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, "")
        
        # Очищаем повторяющиеся знаки препинания
        text = text.replace("...", ".")
        text = text.replace("!!", "!")
        text = text.replace("??", "?")
        text = text.replace(",,", ",")
        
        # Удаляем повторяющиеся пробелы после очистки
        text = " ".join(text.split())
        
        self.logger.debug(f"Постобработка текста: '{text}'")
        return text.strip()
    
    async def transcribe_telegram_voice(self, bot, voice_message, user_id: int) -> Optional[str]:
        """
        Специальный метод для обработки голосовых сообщений Telegram.
        
        Args:
            bot: Экземпляр бота Telegram
            voice_message: Объект голосового сообщения
            user_id: ID пользователя
            
        Returns:
            Optional[str]: Распознанный текст
        """
        try:
            # Проверяем длительность сообщения
            if voice_message.duration > config.security.max_voice_duration:
                self.logger.warning(
                    f"⚠️ Голосовое сообщение слишком длинное: {voice_message.duration}s "
                    f"(макс. {config.security.max_voice_duration}s)"
                )
                return None
            
            # Проверяем размер файла
            if voice_message.file_size > config.file.max_file_size:
                self.logger.warning(
                    f"⚠️ Голосовое сообщение слишком большое: {voice_message.file_size} bytes "
                    f"(макс. {config.file.max_file_size} bytes)"
                )
                return None
            
            # Скачиваем файл
            file = await bot.get_file(voice_message.file_id)
            
            # Создаем имя временного файла
            filename = f"voice_{user_id}_{int(time.time())}.ogg"
            audio_path = config.get_temp_audio_path(filename)
            
            self.logger.info(f"📥 Скачивание голосового файла: {filename}")
            self.logger.debug(f"   File ID: {voice_message.file_id}")
            self.logger.debug(f"   Размер: {voice_message.file_size} bytes")
            self.logger.debug(f"   Длительность: {voice_message.duration}s")
            
            # Скачиваем аудиофайл
            await bot.download_file(file.file_path, audio_path)
            
            self.logger.info(f"✅ Голосовой файл скачан: {audio_path}")
            
            # Транскрибируем с использованием пула
            result = await self.transcribe_audio(audio_path, user_id)
            
            # Удаляем временный файл
            try:
                Path(audio_path).unlink()
                self.logger.debug(f"Удален временный файл: {audio_path}")
            except Exception as e:
                self.logger.warning(f"Не удалось удалить временный файл {audio_path}: {e}")
            
            return result
            
        except Exception as e:
            self.logger.error(
                f"error={e},"
                f"context={{"
                f'"method": "transcribe_telegram_voice",'
                f'"user_id": {user_id},'
                f'"voice_duration": {getattr(voice_message, "duration", "unknown")},'
                f'"voice_file_id": {getattr(voice_message, "file_id", "unknown")},'
                f'"voice_file_size": {getattr(voice_message, "file_size", "unknown")}'
            f"}}"
            )
            return None
    
    def get_model_info(self) -> dict:
        """
        Получение информации о загруженной модели.
        
        Returns:
            dict: Информация о модели
        """
        model_info = {
            "model_name": self.model_name,
            "language": self.language,
            "gpu_devices": config.speech.gpu_devices,
            "whisper_pool_status": self.whisper_pool.get_status(),
        }
        
        # Добавляем информацию о доступных языках
        try:
            if hasattr(whisper, 'tokenizer') and hasattr(whisper.tokenizer, 'LANGUAGES'):
                model_info["supported_languages"] = list(whisper.tokenizer.LANGUAGES.keys())
            else:
                # Fallback для старых версий
                model_info["supported_languages"] = ["ru", "en", "es", "fr", "de", "it", "pt", "zh", "ja", "ko"]
        except Exception:
            model_info["supported_languages"] = ["unknown"]
        
        # Добавляем информацию о доступных моделях
        try:
            model_info["available_models"] = whisper.available_models()
        except Exception:
            model_info["available_models"] = ["unknown"]
        
        return model_info
    
    async def check_model_health(self) -> bool:
        """
        Проверка работоспособности модели.
        
        Returns:
            bool: True если модель работает корректно
        """
        try:
            self.logger.info("🔍 Проверка работоспособности Whisper пула...")
            
            # Инициализируем пул если нужно
            if not self.whisper_pool._initialized:
                await self.whisper_pool.initialize()
            
            # Проверяем доступность устройств
            status = self.whisper_pool.get_status()
            if status["total_devices"] > 0 and status["initialized"]:
                self.logger.info("✅ Проверка работоспособности Whisper пула прошла успешно")
                return True
            else:
                self.logger.error("❌ Whisper пул не инициализирован или нет доступных устройств")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки работоспособности Whisper пула: {e}")
            return False
    
    def get_status(self) -> dict:
        """
        Получение статуса модуля распознавания речи.
        
        Returns:
            dict: Статус модуля
        """
        try:
            whisper_status = self.whisper_pool.get_status()
            
            return {
                "whisper_available": True,
                "model_name": self.model_name,
                "language": self.language,
                "gpu_devices": config.speech.gpu_devices,
                "whisper_pool_status": whisper_status,
                "audio_processor_available": self.audio_processor.check_ffmpeg_availability(),
                "max_audio_duration": config.speech.max_audio_duration,
                "supported_formats": config.speech.supported_formats
            }
        except Exception as e:
            return {
                "whisper_available": False,
                "error": str(e)
            }
    
    def __del__(self):
        """Деструктор для очистки ресурсов."""
        try:
            # Очистка ресурсов происходит автоматически через пул
            self.logger.debug("✅ SpeechToText ресурсы освобождены")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Ошибка при освобождении ресурсов SpeechToText: {e}")
                