"""
Модуль обработки аудиофайлов для Birthday Bot.
Использует imageio-ffmpeg для конвертации и обработки аудио.
"""

import os
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.config import config
from src.utils.logger import get_speech_logger

class AudioProcessor:
    """Класс для обработки аудиофайлов."""
    
    def __init__(self):
        """Инициализация обработчика аудио."""
        self.logger = get_speech_logger()
        self.ffmpeg_path = self._get_ffmpeg_path()
        
        self.logger.info(f"🎵 Инициализация AudioProcessor")
        if self.ffmpeg_path:
            self.logger.info(f"   FFmpeg: {self.ffmpeg_path}")
        else:
            self.logger.warning("   FFmpeg: недоступен")
    
    def _get_ffmpeg_path(self) -> Optional[str]:
        """
        Получение пути к FFmpeg из пакета imageio-ffmpeg.
        
        Returns:
            Optional[str]: Путь к FFmpeg или None при ошибке
        """
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            
            # Проверяем работоспособность FFmpeg
            result = subprocess.run(
                [ffmpeg_path, "-version"], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Извлекаем версию FFmpeg из вывода
                version_line = result.stdout.split('\n')[0]
                self.logger.info(f"✅ FFmpeg найден: {version_line}")
                return ffmpeg_path
            else:
                self.logger.error(f"❌ FFmpeg не работает: {result.stderr}")
                return None
                
        except ImportError:
            self.logger.error("❌ imageio-ffmpeg не установлен. Установите: pip install imageio-ffmpeg")
            return None
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Timeout при проверке FFmpeg")
            return None
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения пути к FFmpeg: {e}")
            return None
    
    async def get_audio_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации об аудиофайле.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Optional[Dict]: Информация об аудиофайле
        """
        if not self.ffmpeg_path:
            self.logger.error("❌ FFmpeg недоступен для анализа аудио")
            return None
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", file_path,
                "-hide_banner",
                "-f", "null",
                "-"
            ]
            
            self.logger.debug(f"🔍 Анализ аудиофайла: {file_path}")
            
            # Запускаем команду асинхронно
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Парсим информацию из вывода FFmpeg
            info = {
                "format": None,
                "codec": None,
                "duration": None,
                "duration_seconds": 0.0,
                "channels": None,
                "sample_rate": None,
                "bitrate": None,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            
            # Извлекаем информацию о контейнере/формате
            if "Input #0" in stderr_text:
                input_lines = [line for line in stderr_text.splitlines() if "Input #0" in line]
                if input_lines:
                    input_line = input_lines[0]
                    if ", " in input_line:
                        format_part = input_line.split(", ")[0]
                        if "from" in format_part:
                            info["format"] = format_part.split("from")[0].strip().split()[-1]
            
            # Извлекаем кодек и параметры аудио
            if "Audio:" in stderr_text:
                audio_lines = [line for line in stderr_text.splitlines() if "Audio:" in line]
                if audio_lines:
                    audio_line = audio_lines[0]
                    parts = audio_line.split("Audio:")[1].strip().split(",")
                    
                    if parts:
                        info["codec"] = parts[0].strip()
                    
                    # Частота дискретизации
                    for part in parts:
                        if "Hz" in part:
                            try:
                                sample_rate_str = part.strip().split()[0]
                                info["sample_rate"] = int(sample_rate_str)
                            except (ValueError, IndexError):
                                pass
                    
                    # Количество каналов
                    if "mono" in audio_line.lower():
                        info["channels"] = 1
                    elif "stereo" in audio_line.lower():
                        info["channels"] = 2
                    else:
                        # Пытаемся найти число каналов
                        for part in parts:
                            if "channel" in part.lower():
                                try:
                                    channels_str = part.strip().split()[0]
                                    info["channels"] = int(channels_str)
                                except (ValueError, IndexError):
                                    pass
                    
                    # Битрейт
                    for part in parts:
                        if "kb/s" in part.lower():
                            try:
                                bitrate_str = part.strip().split()[0]
                                info["bitrate"] = f"{bitrate_str} kb/s"
                            except (ValueError, IndexError):
                                pass
            
            # Извлекаем длительность
            if "Duration:" in stderr_text:
                duration_lines = [line for line in stderr_text.splitlines() if "Duration:" in line]
                if duration_lines:
                    duration_line = duration_lines[0]
                    try:
                        duration_part = duration_line.split("Duration:")[1].split(",")[0].strip()
                        info["duration"] = duration_part
                        
                        # Конвертируем в секунды
                        time_parts = duration_part.split(":")
                        if len(time_parts) >= 3:
                            hours = float(time_parts[0])
                            minutes = float(time_parts[1])
                            seconds = float(time_parts[2])
                            info["duration_seconds"] = hours * 3600 + minutes * 60 + seconds
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Ошибка парсинга длительности: {e}")
            
            # Извлекаем общий битрейт если не нашли в аудио потоке
            if not info["bitrate"] and "bitrate:" in stderr_text.lower():
                for line in stderr_text.splitlines():
                    if "bitrate:" in line.lower():
                        try:
                            bitrate_part = line.lower().split("bitrate:")[1].strip().split()[0]
                            info["bitrate"] = bitrate_part
                        except IndexError:
                            pass
                        break
            
            self.logger.debug(f"✅ Информация об аудио получена: {info}")
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации об аудио {file_path}: {e}")
            return None
    
    async def convert_audio(self, input_file: str, output_file: str = None, 
                          sample_rate: int = 16000, channels: int = 1) -> Optional[str]:
        """
        Конвертация аудиофайла в оптимальный формат для Whisper.
        
        Args:
            input_file: Путь к входному файлу
            output_file: Путь к выходному файлу (если None, генерируется автоматически)
            sample_rate: Частота дискретизации
            channels: Количество каналов
            
        Returns:
            Optional[str]: Путь к сконвертированному файлу
        """
        if not self.ffmpeg_path:
            self.logger.error("❌ FFmpeg недоступен для конвертации")
            return None
        
        # Генерируем имя выходного файла если не указано
        if output_file is None:
            input_path = Path(input_file)
            timestamp = int(time.time())
            output_file = str(input_path.parent / f"{input_path.stem}_converted_{timestamp}.wav")
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-y",                    # Перезаписывать существующие файлы
                "-i", input_file,        # Входной файл
                "-ar", str(sample_rate), # Частота дискретизации
                "-ac", str(channels),    # Количество каналов
                "-c:a", "pcm_s16le",    # 16-bit PCM кодирование
                "-f", "wav",            # Формат WAV
                "-loglevel", "error",   # Минимальный вывод
                output_file             # Выходной файл
            ]
            
            self.logger.info(f"🔄 Конвертация аудио: {Path(input_file).name} -> {Path(output_file).name}")
            self.logger.debug(f"   Параметры: {sample_rate}Hz, {channels} канал(ов)")
            self.logger.debug(f"   Команда: {' '.join(cmd)}")
            
            start_time = time.time()
            
            # Запускаем конвертацию асинхронно
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            conversion_time = time.time() - start_time
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace')
                self.logger.error(f"❌ Ошибка FFmpeg конвертации: {error_msg}")
                return None
            
            # Проверяем, что файл создан и не пустой
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                output_size = os.path.getsize(output_file)
                self.logger.info(f"✅ Конвертация завершена за {conversion_time:.2f}с")
                self.logger.debug(f"   Выходной файл: {output_file}")
                self.logger.debug(f"   Размер: {output_size} bytes")
                return output_file
            else:
                self.logger.error(f"❌ Выходной файл пуст или не создан: {output_file}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка конвертации аудио: {e}")
            return None
    
    async def prepare_for_whisper(self, audio_path: str) -> Optional[str]:
        """
        Подготовка аудиофайла для оптимальной работы с Whisper.
        
        Args:
            audio_path: Путь к исходному аудиофайлу
            
        Returns:
            Optional[str]: Путь к подготовленному файлу
        """
        try:
            # Получаем информацию об исходном файле
            audio_info = await self.get_audio_info(audio_path)
            
            if not audio_info:
                self.logger.warning(f"⚠️ Не удалось получить информацию об аудио: {audio_path}")
                return audio_path
            
            # Проверяем, нужна ли конвертация
            needs_conversion = False
            conversion_reasons = []
            
            # Оптимальные параметры для Whisper
            target_sample_rate = 16000
            target_channels = 1
            target_codec = "pcm_s16le"
            
            current_sample_rate = audio_info.get("sample_rate")
            current_channels = audio_info.get("channels")
            current_codec = audio_info.get("codec")
            
            if current_sample_rate != target_sample_rate:
                needs_conversion = True
                conversion_reasons.append(f"частота {current_sample_rate}Hz -> {target_sample_rate}Hz")
            
            if current_channels != target_channels:
                needs_conversion = True
                conversion_reasons.append(f"каналы {current_channels} -> {target_channels}")
            
            if current_codec and target_codec not in current_codec.lower():
                needs_conversion = True
                conversion_reasons.append(f"кодек {current_codec} -> {target_codec}")
            
            # Если конвертация не нужна, возвращаем исходный файл
            if not needs_conversion:
                self.logger.debug(f"✅ Конвертация не требуется: {audio_path}")
                return audio_path
            
            self.logger.info(f"🔧 Требуется конвертация: {', '.join(conversion_reasons)}")
            
            # Выполняем конвертацию
            converted_path = await self.convert_audio(
                input_file=audio_path,
                sample_rate=target_sample_rate,
                channels=target_channels
            )
            
            if converted_path:
                self.logger.info(f"✅ Аудио подготовлено для Whisper: {Path(converted_path).name}")
                return converted_path
            else:
                self.logger.warning(f"⚠️ Конвертация не удалась, используем исходный файл: {audio_path}")
                return audio_path
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка подготовки аудио для Whisper: {e}")
            return audio_path
    
    async def validate_audio_file(self, file_path: str) -> bool:
        """
        Валидация аудиофайла.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            bool: True если файл валиден
        """
        try:
            # Проверяем существование файла
            if not os.path.exists(file_path):
                self.logger.error(f"❌ Аудиофайл не существует: {file_path}")
                return False
            
            # Проверяем размер файла
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                self.logger.error(f"❌ Аудиофайл пуст: {file_path}")
                return False
            
            if file_size > config.file.max_file_size:
                self.logger.error(f"❌ Аудиофайл слишком большой: {file_size} bytes (макс. {config.file.max_file_size} bytes)")
                return False
            
            # Проверяем расширение файла
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in config.speech.supported_formats:
                self.logger.error(f"❌ Неподдерживаемый формат файла: {file_extension}")
                return False
            
            # Получаем информацию об аудио
            audio_info = await self.get_audio_info(file_path)
            if not audio_info:
                self.logger.error(f"❌ Не удалось получить информацию об аудио: {file_path}")
                return False
            
            # Проверяем длительность
            duration = audio_info.get("duration_seconds", 0)
            if duration > config.speech.max_audio_duration:
                self.logger.error(f"❌ Аудио слишком длинное: {duration}s (макс. {config.speech.max_audio_duration}s)")
                return False
            
            if duration == 0:
                self.logger.error(f"❌ Аудио имеет нулевую длительность: {file_path}")
                return False
            
            # Проверяем, что это действительно аудиофайл
            if not audio_info.get("codec"):
                self.logger.error(f"❌ Файл не содержит аудио потока: {file_path}")
                return False
            
            self.logger.debug(f"✅ Аудиофайл валиден: {Path(file_path).name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка валидации аудиофайла {file_path}: {e}")
            return False
    
    def check_ffmpeg_availability(self) -> bool:
        """
        Проверка доступности FFmpeg.
        
        Returns:
            bool: True если FFmpeg доступен
        """
        return self.ffmpeg_path is not None
    
    def get_supported_formats(self) -> list:
        """
        Получение списка поддерживаемых форматов.
        
        Returns:
            list: Список поддерживаемых расширений файлов
        """
        return config.speech.supported_formats
    
    async def cleanup_temp_files(self, max_age_seconds: int = None):
        """
        Очистка временных аудиофайлов.
        
        Args:
            max_age_seconds: Максимальный возраст файлов в секундах
        """
        if max_age_seconds is None:
            max_age_seconds = config.file.max_temp_file_age
        
        try:
            temp_dir = Path(config.speech.temp_audio_dir)
            if not temp_dir.exists():
                self.logger.debug(f"Временная директория не существует: {temp_dir}")
                return
            
            current_time = time.time()
            cleaned_count = 0
            total_size_cleaned = 0
            
            # Очищаем файлы по маске
            patterns = ["voice_*.ogg", "*_converted_*.wav", "temp_audio_*.*"]
            
            for pattern in patterns:
                for file_path in temp_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age > max_age_seconds:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                cleaned_count += 1
                                total_size_cleaned += file_size
                                self.logger.debug(f"Удален старый файл: {file_path.name} ({file_size} bytes)")
                        except Exception as e:
                            self.logger.warning(f"Не удалось удалить файл {file_path}: {e}")
            
            if cleaned_count > 0:
                size_mb = total_size_cleaned / (1024 * 1024)
                self.logger.info(f"🧹 Очищено {cleaned_count} временных аудиофайлов ({size_mb:.2f} MB)")
            else:
                self.logger.debug("Нет старых временных файлов для очистки")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка очистки временных файлов: {e}")
    
    def get_status(self) -> dict:
        """
        Получение статуса аудиопроцессора.
        
        Returns:
            dict: Статус компонентов
        """
        return {
            "ffmpeg_available": self.check_ffmpeg_availability(),
            "ffmpeg_path": self.ffmpeg_path,
            "supported_formats": self.get_supported_formats(),
            "max_file_size": config.file.max_file_size,
            "max_audio_duration": config.speech.max_audio_duration,
            "temp_dir": config.speech.temp_audio_dir
        }
    
    async def get_detailed_audio_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Детальный анализ аудиофайла для отладки.
        
        Args:
            file_path: Путь к аудиофайлу
            
        Returns:
            Optional[Dict]: Детальная информация об аудиофайле
        """
        if not self.ffmpeg_path:
            return None
        
        try:
            cmd = [
                self.ffmpeg_path,
                "-i", file_path,
                "-hide_banner",
                "-f", "null",
                "-v", "info",
                "-"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            # Базовая информация
            basic_info = await self.get_audio_info(file_path)
            if not basic_info:
                return None
            
            # Дополнительная информация
            detailed_info = basic_info.copy()
            detailed_info.update({
                "ffmpeg_output": stderr_text,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "file_extension": Path(file_path).suffix,
                "analysis_timestamp": time.time()
            })
            
            return detailed_info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка детального анализа аудио: {e}")
            return None
            