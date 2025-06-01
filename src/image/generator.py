"""
Модуль генерации изображений с использованием локальных Stable Diffusion моделей.
Создает поздравительные картинки на основе текста с AI-генерацией.
"""

import asyncio
import time
import gc
import re
import shutil
from pathlib import Path
from typing import Optional, List
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import config
from src.utils.logger import get_image_logger
from transformers import MarianMTModel, MarianTokenizer

# Инициализация логгера
logger = get_image_logger()

class ImageGenerator:
    """Генератор поздравительных изображений с локальными AI моделями."""
    
    def __init__(self, progress_callback=None):
        """
        Инициализация генератора.
        
        Args:
            progress_callback: Функция для отправки сообщений о прогрессе
        """
        self.device = None
        self.pipeline = None
        self.model_loaded = False
        self.progress_callback = progress_callback
        
        logger.info("🎨 Инициализирован локальный генератор изображений")
        logger.info(f"   Модель: {config.diffusion.model}")
        logger.info(f"   Количество изображений: {config.diffusion.num_images}")
        
        # Инициализируем устройство
        self.device = self._get_device()
        logger.info(f"   Устройство: {self.device}")
        
        # Проверяем зависимости
        if not self._check_dependencies():
            raise ImportError("Не установлены необходимые зависимости для локальной генерации изображений")

    def _get_device(self):
        """Определение оптимального устройства для вычислений."""
        try:
            import torch
        except ImportError:
            logger.error("❌ PyTorch не установлен")
            return "cpu"
            
        if config.diffusion.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                try:
                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"🚀 Обнаружена CUDA GPU: {device_name}")
                except:
                    logger.info("🚀 CUDA доступна")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("🍎 Использование Apple Silicon GPU (MPS)")
            else:
                device = "cpu"
                logger.info("💻 Использование CPU")
        else:
            device = config.diffusion.device
            
        return device

    def _get_expected_model_load_time(self) -> int:
        """Получение ожидаемого времени загрузки модели в секундах."""
        model_name = config.diffusion.model.lower()
        
        if "flux" in model_name:
            if self.device == "cuda":
                return 15  # FLUX на GPU
            else:
                return 40  # FLUX на CPU
        elif "xl" in model_name:
            if self.device == "cuda":
                return 30  # SDXL на GPU
            else:
                return 90   # SDXL на CPU
        else:
            if self.device == "cuda":
                return 20   # SD на GPU
            else:
                return 60   # SD на CPU

    def _get_expected_translation_time(self) -> int:
        """Получение ожидаемого времени перевода текста в секундах."""
        return 2

    def _get_expected_generation_time(self) -> int:
        """Получение ожидаемого времени генерации изображений в секундах."""
        model_name = config.diffusion.model.lower()
        num_images = config.diffusion.num_images
        steps = config.diffusion.num_inference_steps
        
        # Базовое время на одно изображение
        if "flux" in model_name:
            if self.device == "cuda":
                base_time = 12  # FLUX на GPU - быстрее
            else:
                base_time = 50  # FLUX на CPU
        elif "xl" in model_name:
            if self.device == "cuda":
                base_time = 8   # SDXL на GPU
            else:
                base_time = 45  # SDXL на CPU
        else:
            if self.device == "cuda":
                base_time = 5   # SD на GPU
            else:
                base_time = 30  # SD на CPU
        
        # Учитываем количество шагов и изображений
        time_per_image = base_time * (steps / 20)  # Нормализуем к 20 шагам
        total_time = time_per_image * num_images
        
        return int(total_time)

    async def _send_progress_message(self, message_key: str, **kwargs):
        """Отправка сообщения о прогрессе через callback."""
        if self.progress_callback:
            try:
                await self.progress_callback(message_key, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ Ошибка отправки сообщения о прогрессе: {e}")

    def _check_dependencies(self):
        """Проверка необходимых зависимостей."""
        try:
            import torch
            import diffusers
            logger.info("✅ Все зависимости для локальной генерации доступны")
            return True
        except ImportError as e:
            logger.error(f"❌ Отсутствуют зависимости: {e}")
            logger.error("Установите: pip install torch diffusers transformers")
            return False

    async def generate_birthday_image(self, text: str, user_id: int) -> Optional[str]:
        """
        Генерация поздравительных картинок.
        
        Args:
            text: Текст поздравления
            user_id: ID пользователя
            
        Returns:
            Путь к директории с сгенерированными изображениями или None при ошибке
        """
        start_time = time.time()
        
        try:
            logger.info(f"🎨 Начинаем генерацию {config.diffusion.num_images} изображений для пользователя {user_id}")
            
            # Создаем временную директорию для пользователя
            output_dir = config.create_temp_images_dir(user_id)
            logger.info(f"📁 Создана временная директория: {output_dir}")
            
            images = await self._generate_with_diffusion(text)
            
            if images and len(images) > 0:
                # Сохраняем все изображения в директорию
                saved_paths = []
                for i, image in enumerate(images):
                    if image:
                        filename = f"birthday_card_{i+1}.png"
                        image_path = Path(output_dir) / filename
                        
                        try:
                            image.save(image_path, "PNG", quality=95)
                            saved_paths.append(str(image_path))
                            logger.debug(f"✅ Сохранено изображение {i+1}: {filename}")
                        except Exception as e:
                            logger.error(f"❌ Ошибка сохранения изображения {i+1}: {e}")
                
                if saved_paths:
                    generation_time = time.time() - start_time
                    logger.info(f"✅ Сгенерировано и сохранено {len(saved_paths)} изображений за {generation_time:.2f}с")
                    
                    # Очищаем память
                    self._cleanup_memory()
                    
                    return output_dir
                else:
                    logger.error("❌ Не удалось сохранить ни одного изображения")
                    # Удаляем пустую директорию
                    self._cleanup_directory(output_dir)
                    return None
            else:
                logger.error("❌ Не удалось сгенерировать изображения")
                # Удаляем пустую директорию
                self._cleanup_directory(output_dir)
                return None
                
        except Exception as e:
            logger.error(
                f"error={e},"
                f"context={{"
                f"method: 'generate_birthday_image',"
                f"user_id: {user_id},"
                f"text_length: {len(text)}"
                f"}}"
            )
            # Очищаем директорию при ошибке
            if 'output_dir' in locals():
                self._cleanup_directory(output_dir)
            return None

    def _cleanup_directory(self, directory_path: str) -> None:
        """
        Очистка директории и её удаление.
        
        Args:
            directory_path: Путь к директории для удаления
        """
        try:
            dir_path = Path(directory_path)
            if dir_path.exists() and dir_path.is_dir():
                shutil.rmtree(dir_path)
                logger.debug(f"🗑️ Удалена директория: {directory_path}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось удалить директорию {directory_path}: {e}")

    async def _load_diffusion_model(self):
        """Загрузка локальной diffusion модели."""
        if self.model_loaded:
            return True
            
        try:
            import torch
            from diffusers import (
                StableDiffusionXLPipeline, 
                StableDiffusionPipeline,
                DiffusionPipeline
            )
            
            # Отправляем сообщение о начале загрузки
            await self._send_progress_message(
                "model_loading_start",
                expected_time=self._get_expected_model_load_time()
            )
            
            logger.info("📥 Загрузка локальной diffusion модели...")
            start_time = time.time()
            
            model_name = config.diffusion.model
            
            # Определяем тип pipeline по названию модели
            if "flux" in model_name.lower():
                logger.info("🔧 Загрузка FLUX pipeline...")
                try:
                    from diffusers import FluxPipeline
                    pipeline_class = FluxPipeline
                    
                    # Специальные настройки для FLUX
                    load_kwargs = {
                        "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
                    }
                    
                    # FLUX не поддерживает device_map="auto", используем обычную загрузку
                    logger.info(f"Загрузка FLUX модели {model_name}...")
                    self.pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                    
                except ImportError as e:
                    logger.error(f"error: {e}")
                    logger.error(f"❌ FluxPipeline не найден. Обновите diffusers: pip install diffusers>=0.30.0")
                    import diffusers
                    logger.info(f"diffusers: {diffusers.__version__}")
                    return False
                    
            elif "xl" in model_name.lower():
                logger.info("🔧 Загрузка SDXL pipeline...")
                pipeline_class = StableDiffusionXLPipeline
                
                # Настройки загрузки для SDXL
                load_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                    "safety_checker": None,
                    "requires_safety_checker": False
                }
                
                # Добавляем variant для fp16 моделей
                if self.device != "cpu":
                    load_kwargs["variant"] = "fp16"
                
                # Загружаем модель
                self.pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                
            elif "stable-diffusion" in model_name.lower():
                logger.info("🔧 Загрузка SD pipeline...")
                pipeline_class = StableDiffusionPipeline
                
                # Настройки загрузки для SD
                load_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                    "safety_checker": None,
                    "requires_safety_checker": False
                }
                
                if self.device != "cpu":
                    load_kwargs["variant"] = "fp16"
                
                self.pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                
            else:
                logger.info("🔧 Загрузка универсального pipeline...")
                pipeline_class = DiffusionPipeline
                
                # Настройки загрузки для универсального pipeline
                load_kwargs = {
                    "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                }
                
                self.pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
            
            # Перемещаем на устройство
            self.pipeline = self.pipeline.to(self.device)
            
            # Применяем оптимизации
            self._apply_optimizations()
            
            self.model_loaded = True
            load_time = time.time() - start_time
            
            # Отправляем сообщение о завершении загрузки
            logger.info(f"model_loading_done")
            await self._send_progress_message(
                "model_loading_done",
                actual_time=load_time
            )
            
            logger.info(f"✅ Локальная модель загружена за {load_time:.1f}с")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки локальной модели: {e}")
            return False

    def _apply_optimizations(self):
        """Применение оптимизаций для разных устройств."""
        try:
            if self.device == "cuda":
                # Оптимизации для CUDA
                if hasattr(self.pipeline, 'enable_memory_efficient_attention'):
                    self.pipeline.enable_memory_efficient_attention()
                    
                if config.diffusion.enable_xformers:
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("✅ Включена xformers оптимизация")
                    except Exception:
                        logger.warning("⚠️ xformers недоступен")
                        
                if config.diffusion.enable_cpu_offload:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("✅ Включена CPU разгрузка")
                    
            elif self.device == "mps":
                # Оптимизации для Apple Silicon
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Включена attention slicing для MPS")
                    
            else:  # CPU
                # Оптимизации для CPU
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Включена attention slicing для CPU")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка применения оптимизаций: {e}")

    def _cleanup_memory(self):
        """Очистка памяти после генерации."""
        try:
            import torch
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Ошибка очистки памяти: {e}")

    async def _generate_with_diffusion(self, text: str) -> Optional[List[Image.Image]]:
        """
        Генерация изображений с локальной моделью diffusion.
        
        Args:
            text: Текст поздравления
            
        Returns:
            Список PIL Image или None при ошибке
        """
        # Загружаем модель если нужно
        if not self.model_loaded:
            if not await self._load_diffusion_model():
                logger.error("❌ Не удалось загрузить локальную модель")
                return None
        
        try:
            # Генерируем изображения с локальной моделью

            import torch
            
            logger.info(f"🔄 Генерация {config.diffusion.num_images} изображений с локальной моделью...")
            
            # Создаем промпт для генерации

            await self._send_progress_message(
                "translation_start",
                expected_time=self._get_expected_translation_time()
            )
            translation_start_time = time.time()

            prompt = self._create_birthday_prompt(text)
            logger.info(f"📝 Промпт: {prompt}...")
            
            translation_time = time.time() - translation_start_time
            await self._send_progress_message(
                "translation_done",
                actual_time=translation_time
            )

            await self._send_progress_message(
                "image_generation_start",
                num_images=config.diffusion.num_images,
                expected_time=self._get_expected_generation_time()
            )
            generation_start_time = time.time()

            # Параметры генерации (разные для FLUX и других моделей)
            if "flux" in config.diffusion.model.lower():
                # Параметры для FLUX
                generation_params = {
                    "prompt": prompt,
                    "height": config.diffusion.height,
                    "width": config.diffusion.width,
                    "num_inference_steps": config.diffusion.num_inference_steps,
                    "guidance_scale": config.diffusion.guidance_scale,
                    "num_images_per_prompt": config.diffusion.num_images,
                }
                
                # FLUX может не поддерживать negative_prompt
                logger.debug("Генерация с FLUX моделью...")
                
            else:
                # Параметры для Stable Diffusion
                generation_params = {
                    "prompt": prompt,
                    "height": config.diffusion.height,
                    "width": config.diffusion.width,
                    "num_inference_steps": config.diffusion.num_inference_steps,
                    "guidance_scale": config.diffusion.guidance_scale,
                    "num_images_per_prompt": config.diffusion.num_images,
                }
                
                # Добавляем негативный промпт для SD
                if config.diffusion.negative_prompt:
                    generation_params["negative_prompt"] = config.diffusion.negative_prompt
            
            # Добавляем generator для seed
            if config.diffusion.seed >= 0:
                generation_params["generator"] = torch.Generator().manual_seed(config.diffusion.seed)
            
            logger.debug(f"🎛️ Параметры: steps={generation_params['num_inference_steps']}, "
                        f"guidance={generation_params['guidance_scale']}, "
                        f"images={generation_params['num_images_per_prompt']}")
            
            # Запускаем генерацию в отдельном потоке
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_pipeline,
                generation_params
            )

            generation_time = time.time() - generation_start_time
            await self._send_progress_message(
                "image_generation_done",
                actual_time=generation_time
            )

            # Получаем изображения из результата
            if result and hasattr(result, 'images') and result.images:
                images = result.images
                logger.info(f"✅ Успешно сгенерировано {len(images)} изображений локально")
                return images
            else:
                logger.error("❌ Не удалось получить изображения из результата")
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации с diffusion: {e}")
            return None

    def _run_pipeline(self, params: dict):
        """Запуск pipeline в отдельном потоке."""
        try:
            import torch
            
            with torch.no_grad():  # Экономим память
                return self.pipeline(**params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения pipeline: {e}")
            return None

    def _create_birthday_prompt(self, text: str) -> str:
        """
        Создание промпта для генерации изображения.
        
        Args:
            text: Текст поздравления
            
        Returns:
            Промпт для AI генерации
        """
        # Получаем настройки промптов из конфигурации
        prompts_config = config.diffusion.prompts
        
        # Определяем контент на основе языка текста
        if self.has_russian(text):
            content = self.translate_text(text)
        else:
            content = text
        
        # Формируем промпт по шаблону из конфигурации
        if "style".lower() in content.lower():
            prompt = prompts_config.template_no_style.format(
                picture=prompts_config.base_picture,
                subject=prompts_config.subject,
                content=content,
            )
        else:        
            prompt = prompts_config.template_with_style.format(
                picture=prompts_config.base_picture,
                style=prompts_config.style,
                subject=prompts_config.subject,
                content=content,
            )
        
        return prompt

    def has_russian(self, text):
        return bool(re.search(r"[а-яА-ЯёЁ]", text))

    def translate_text(self, text: str) -> str:
        # Переводим текст на английский
        model_name = "Helsinki-NLP/opus-mt-ru-en"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return result

    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """
        Очистка старых временных файлов и директорий.
        
        Args:
            max_age_hours: Максимальный возраст файлов в часах
        """
        try:
            temp_dir = Path(config.paths.temp_images)
            if not temp_dir.exists():
                return
            
            max_age_seconds = max_age_hours * 3600
            current_time = time.time()
            
            cleaned_count = 0
            
            # Очищаем старые директории с изображениями
            for dir_path in temp_dir.glob("birthday_cards_*"):
                if dir_path.is_dir():
                    if current_time - dir_path.stat().st_mtime > max_age_seconds:
                        try:
                            shutil.rmtree(dir_path)
                            cleaned_count += 1
                            logger.debug(f"🗑️ Удалена старая директория: {dir_path.name}")
                        except Exception as e:
                            logger.warning(f"⚠️ Не удалось удалить директорию {dir_path}: {e}")
            
            # Очищаем отдельные старые файлы (для совместимости)
            for file_path in temp_dir.glob("birthday_card_*.png"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"🗑️ Удален старый файл: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Не удалось удалить файл {file_path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"🧹 Очищено {cleaned_count} старых элементов")
                
        except Exception as e:
            logger.error(f"❌ Ошибка очистки временных файлов: {e}")

    def get_image_paths_from_dir(self, directory_path: str) -> List[str]:
        """
        Получение путей ко всем изображениям в директории.
        
        Args:
            directory_path: Путь к директории
            
        Returns:
            Список путей к изображениям
        """
        try:
            dir_path = Path(directory_path)
            if not dir_path.exists() or not dir_path.is_dir():
                logger.warning(f"⚠️ Директория не существует: {directory_path}")
                return []
            
            # Ищем PNG файлы с названием birthday_card_*
            image_paths = []
            for i in range(1, config.diffusion.num_images + 1):
                image_path = dir_path / f"birthday_card_{i}.png"
                if image_path.exists():
                    image_paths.append(str(image_path))
                else:
                    logger.warning(f"⚠️ Изображение не найдено: {image_path}")
            
            logger.debug(f"📁 Найдено {len(image_paths)} изображений в {directory_path}")
            return image_paths
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения путей к изображениям: {e}")
            return []

    def get_status(self) -> dict:
        """
        Получение статуса генератора изображений.
        
        Returns:
            dict: Статус различных компонентов
        """
        return {
            "local_diffusion_available": self._check_dependencies(),
            "local_model": config.diffusion.model,
            "local_device": self.device,
            "local_model_loaded": self.model_loaded,
            "num_images_per_generation": config.diffusion.num_images,
            "dependencies_installed": self._check_dependencies()
        }
        