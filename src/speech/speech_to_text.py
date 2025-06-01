"""
Модуль распознавания речи для Birthday Bot.
Использует OpenAI Whisper для конвертации голосовых сообщений в текст.
"""

import os
import asyncio
import time
import whisper
from pathlib import Path
from typing import Optional

from src.utils.config import config
from src.utils.logger import get_speech_logger
from src.speech.audio_processor import AudioProcessor

class SpeechToText:
    """Класс для распознавания речи с использованием Whisper."""
    
    def __init__(self):
        """Инициализация модуля распознавания речи."""
        self.logger = get_speech_logger()
        self.model = None
        self.audio_processor = AudioProcessor()
        
        # Используем правильные поля из конфигурации
        self.model_name = config.speech.model_name
        self.device = config.speech.device
        self.language = config.speech.language
        
        self.logger.info(f"🎤 Инициализация SpeechToText с моделью: {self.model_name}")
        self.logger.info(f"   Устройство: {self.device}")
        self.logger.info(f"   Язык: {self.language}")
        
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
    
    def _load_model(self):
        """
        Загружает модель Whisper (lazy loading).
        """
        if self.model is None:
            try:
                self.logger.info(f"📥 Загрузка модели Whisper: {self.model_name}")
                start_time = time.time()
                
                self.model = whisper.load_model(
                    self.model_name,
                    device=self.device
                )
                
                load_time = time.time() - start_time
                self.logger.info(f"✅ Модель Whisper загружена за {load_time:.2f}с")
                
                # Логируем информацию о модели
                model_info = self.get_model_info()
                self.logger.debug(f"   Информация о модели: {model_info}")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка загрузки модели Whisper: {e}")
                raise
    
    async def transcribe_audio(self, audio_path: str, user_id: int = None) -> Optional[str]:
        """
        Транскрибирует аудиофайл в текст.
        
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
            
            # Загружаем модель если еще не загружена
            self._load_model()
            
            # Запускаем транскрибацию в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._transcribe_sync, processed_audio_path)
            
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
    
    def _transcribe_sync(self, audio_path: str) -> Optional[str]:
        """
        Синхронная транскрибация аудио (выполняется в отдельном потоке).
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Optional[str]: Распознанный текст
        """
        try:
            self.logger.debug(f"🔄 Выполнение синхронной транскрибации: {audio_path}")
            
            result = self.model.transcribe(
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
            
            # Транскрибируем
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
            "device": self.device,
            "language": self.language,
            "is_loaded": self.model is not None,
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
            self.logger.info("🔍 Проверка работоспособности модели Whisper...")
            
            # Загружаем модель если не загружена
            self._load_model()
            
            # Создаем тестовый аудиофайл (тишина)
            import numpy as np
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 секунда тишины
            
            # Тестируем модель
            result = self.model.transcribe(test_audio, language=self.language)
            
            # Проверяем, что модель вернула результат
            if "text" in result:
                self.logger.info("✅ Проверка работоспособности модели прошла успешно")
                return True
            else:
                self.logger.error("❌ Модель не вернула ожидаемый результат")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки работоспособности модели: {e}")
            return False
    
    def get_status(self) -> dict:
        """
        Получение статуса модуля распознавания речи.
        
        Returns:
            dict: Статус модуля
        """
        try:
            return {
                "whisper_available": True,
                "model_name": self.model_name,
                "model_loaded": self.model is not None,
                "device": self.device,
                "language": self.language,
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
        if hasattr(self, 'model') and self.model is not None:
            try:
                # Освобождаем память модели
                del self.model
                self.logger.debug("✅ Модель Whisper выгружена из памяти")
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Ошибка при выгрузке модели: {e}")
                    