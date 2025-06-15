"""
Модуль генерации изображений с использованием локальных Stable Diffusion моделей.
Создает поздравительные картинки на основе текста с AI-генерацией.
Поддержка multi-GPU для параллельной генерации и перевода.
"""

import asyncio
import time
import gc
import re
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from contextlib import asynccontextmanager

from src.utils.config import config
from src.utils.logger import get_image_logger
from transformers import MarianMTModel, MarianTokenizer

# Инициализация логгера
logger = get_image_logger()

class TranslatorPool:
    """Пул переводчиков для параллельного перевода текста."""
    
    def __init__(self, gpu_devices: List[str]):
        """
        Инициализация пула переводчиков.
        
        Args:
            gpu_devices: Список GPU устройств
        """
        self.gpu_devices = gpu_devices
        self.model_name = "Helsinki-NLP/opus-mt-ru-en"
        self.tokenizers: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.available_devices = asyncio.Queue(maxsize=len(gpu_devices))
        self._initialized = False
        
        logger.info(f"🔤 Инициализирован пул переводчиков с {len(gpu_devices)} устройствами: {gpu_devices}")
    
    async def initialize(self):
        """Инициализация всех моделей перевода для GPU."""
        if self._initialized:
            return
        
        logger.info("📥 Загрузка моделей перевода на все устройства...")
        
        for device in self.gpu_devices:
            try:
                tokenizer, model = await self._load_translator_for_device(device)
                if tokenizer and model:
                    self.tokenizers[device] = tokenizer
                    self.models[device] = model
                    await self.available_devices.put(device)
                    logger.info(f"✅ Модель перевода загружена для {device}")
                else:
                    logger.error(f"❌ Не удалось загрузить модель перевода для {device}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели перевода для {device}: {e}")
        
        self._initialized = True
        logger.info(f"🚀 Пул переводчиков инициализирован с {len(self.models)} активными устройствами")
    
    async def _load_translator_for_device(self, device: str):
        """Загрузка модели перевода для конкретного устройства."""
        try:
            import warnings
            warnings.filterwarnings("ignore", message=".*add_prefix_space.*")
            
            # Загружаем в отдельном потоке
            loop = asyncio.get_event_loop()
            
            tokenizer = await loop.run_in_executor(
                None,
                MarianTokenizer.from_pretrained,
                self.model_name
            )
            
            model = await loop.run_in_executor(
                None,
                MarianMTModel.from_pretrained,
                self.model_name
            )
            
            # Перемещаем на устройство
            if device != "cpu":
                model = model.to(device)
            
            return tokenizer, model
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки переводчика для {device}: {e}")
            return None, None
    
    @asynccontextmanager
    async def acquire_translator(self):
        """Контекстный менеджер для получения переводчика из пула."""
        if not self._initialized:
            await self.initialize()
        
        # Ждем свободное устройство
        device = await self.available_devices.get()
        tokenizer = self.tokenizers.get(device)
        model = self.models.get(device)
        
        if not tokenizer or not model:
            await self.available_devices.put(device)
            raise RuntimeError(f"Переводчик для {device} недоступен")
        
        try:
            logger.debug(f"🔒 Получен доступ к переводчику {device}")
            yield device, tokenizer, model
        finally:
            # Очищаем память и возвращаем устройство в пул
            self._cleanup_device_memory(device)
            await self.available_devices.put(device)
            logger.debug(f"🔓 Освобожден переводчик {device}")
    
    def _cleanup_device_memory(self, device: str):
        """Очистка памяти конкретного устройства."""
        try:
            import torch
            
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            elif device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Ошибка очистки памяти переводчика {device}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса пула переводчиков."""
        return {
            "total_devices": len(self.gpu_devices),
            "available_devices": self.available_devices.qsize(),
            "busy_devices": len(self.gpu_devices) - self.available_devices.qsize(),
            "model_name": self.model_name,
            "initialized": self._initialized
        }

# Глобальный пул переводчиков (синглтон)
_translator_pool: Optional[TranslatorPool] = None

def get_translator_pool() -> TranslatorPool:
    """Получение глобального пула переводчиков."""
    global _translator_pool
    if _translator_pool is None:
        # Используем те же GPU что и для генерации изображений
        gpu_devices = config.diffusion.gpu_devices
        if not gpu_devices:
            gpu_devices = ["cpu"]  # Fallback
        _translator_pool = TranslatorPool(gpu_devices)
    return _translator_pool

class GPUPool:
    """Пул GPU для параллельной генерации изображений."""
    
    def __init__(self, gpu_devices: List[str]):
        """
        Инициализация пула GPU.
        
        Args:
            gpu_devices: Список GPU устройств
        """
        self.gpu_devices = gpu_devices
        self.pipelines: Dict[str, Any] = {}
        self.available_gpus = asyncio.Queue(maxsize=len(gpu_devices))
        self.generation_queue = asyncio.Queue(maxsize=config.diffusion.max_queue_size)
        self._initialized = False
        
        logger.info(f"🎮 Инициализирован GPU пул с {len(gpu_devices)} устройствами: {gpu_devices}")
    
    async def initialize(self):
        """Инициализация всех pipeline для GPU."""
        if self._initialized:
            return
        
        logger.info("📥 Загрузка моделей на все GPU...")
        
        for device in self.gpu_devices:
            try:
                pipeline = await self._load_pipeline_for_device(device)
                if pipeline:
                    self.pipelines[device] = pipeline
                    await self.available_gpus.put(device)
                    logger.info(f"✅ Pipeline загружен для {device}")
                else:
                    logger.error(f"❌ Не удалось загрузить pipeline для {device}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки pipeline для {device}: {e}")
        
        self._initialized = True
        logger.info(f"🚀 GPU пул инициализирован с {len(self.pipelines)} активными устройствами")
    
    async def _load_pipeline_for_device(self, device: str):
        """Загрузка pipeline для конкретного устройства."""
        try:
            import torch
            from diffusers import (
                StableDiffusionXLPipeline, 
                StableDiffusionPipeline,
                DiffusionPipeline
            )
            
            model_name = config.diffusion.model
            
            # Определяем тип pipeline по названию модели
            if "flux" in model_name.lower():
                try:
                    from diffusers import FluxPipeline
                    pipeline_class = FluxPipeline
                    
                    load_kwargs = {
                        "torch_dtype": torch.bfloat16 if device != "cpu" else torch.float32,
                    }
                    
                    pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                    
                except ImportError:
                    logger.error(f"❌ FluxPipeline не найден для {device}")
                    return None
                    
            elif "xl" in model_name.lower():
                pipeline_class = StableDiffusionXLPipeline
                
                load_kwargs = {
                    "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
                    "safety_checker": None,
                    "requires_safety_checker": False
                }
                
                if device != "cpu":
                    load_kwargs["variant"] = "fp16"
                
                pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                
            elif "stable-diffusion" in model_name.lower():
                pipeline_class = StableDiffusionPipeline
                
                load_kwargs = {
                    "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
                    "safety_checker": None,
                    "requires_safety_checker": False
                }
                
                if device != "cpu":
                    load_kwargs["variant"] = "fp16"
                
                pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
                
            else:
                pipeline_class = DiffusionPipeline
                
                load_kwargs = {
                    "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
                }
                
                pipeline = pipeline_class.from_pretrained(model_name, **load_kwargs)
            
            # Перемещаем на устройство
            pipeline = pipeline.to(device)
            
            # Применяем оптимизации
            self._apply_optimizations(pipeline, device)
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки pipeline для {device}: {e}")
            return None
    
    def _apply_optimizations(self, pipeline, device: str):
        """Применение оптимизаций для конкретного устройства."""
        try:
            if device.startswith("cuda"):
                if hasattr(pipeline, 'enable_memory_efficient_attention'):
                    pipeline.enable_memory_efficient_attention()
                    
                if config.diffusion.enable_xformers:
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()
                    except Exception:
                        pass
                        
                if config.diffusion.enable_cpu_offload:
                    pipeline.enable_model_cpu_offload()
                    
            elif device == "mps":
                if hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing()
                    
            else:  # CPU
                if hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing()
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка применения оптимизаций для {device}: {e}")
    
    @asynccontextmanager
    async def acquire_gpu(self):
        """Контекстный менеджер для получения GPU из пула."""
        if not self._initialized:
            await self.initialize()
        
        # Ждем свободную GPU
        device = await self.available_gpus.get()
        pipeline = self.pipelines.get(device)
        
        if not pipeline:
            await self.available_gpus.put(device)
            raise RuntimeError(f"Pipeline для {device} недоступен")
        
        try:
            logger.debug(f"🔒 Получен доступ к {device}")
            yield device, pipeline
        finally:
            # Очищаем память и возвращаем GPU в пул
            self._cleanup_device_memory(device)
            await self.available_gpus.put(device)
            logger.debug(f"🔓 Освобожден {device}")
    
    def _cleanup_device_memory(self, device: str):
        """Очистка памяти конкретного устройства."""
        try:
            import torch
            
            if device.startswith("cuda"):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            elif device == "mps":
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            
            gc.collect()
            
        except Exception as e:
            logger.debug(f"Ошибка очистки памяти {device}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса пула GPU."""
        return {
            "total_gpus": len(self.gpu_devices),
            "available_gpus": self.available_gpus.qsize(),
            "busy_gpus": len(self.gpu_devices) - self.available_gpus.qsize(),
            "queue_size": self.generation_queue.qsize(),
            "max_queue_size": config.diffusion.max_queue_size,
            "initialized": self._initialized
        }

# Глобальный пул GPU (синглтон)
_gpu_pool: Optional[GPUPool] = None

def get_gpu_pool() -> GPUPool:
    """Получение глобального пула GPU."""
    global _gpu_pool
    if _gpu_pool is None:
        gpu_devices = config.diffusion.gpu_devices
        if not gpu_devices:
            gpu_devices = ["cpu"]  # Fallback
        _gpu_pool = GPUPool(gpu_devices)
    return _gpu_pool

class ImageGenerator:
    """Генератор поздравительных изображений с локальными AI моделями."""
    
    def __init__(self, progress_callback=None):
        """
        Инициализация генератора.
        
        Args:
            progress_callback: Функция для отправки сообщений о прогрессе
        """
        self.progress_callback = progress_callback
        self.gpu_pool = get_gpu_pool()
        self.translator_pool = get_translator_pool()
        
        logger.info("🎨 Инициализирован multi-GPU генератор изображений")
        logger.info(f"   Модель: {config.diffusion.model}")
        logger.info(f"   Количество изображений: {config.diffusion.num_images}")
        logger.info(f"   GPU устройства: {config.diffusion.gpu_devices}")
        
        # Проверяем зависимости
        if not self._check_dependencies():
            raise ImportError("Не установлены необходимые зависимости для локальной генерации изображений")

    def _get_expected_translation_time(self) -> int:
        """Получение ожидаемого времени перевода текста в секундах."""
        return 2

    def _get_expected_generation_time(self) -> int:
        """Получение ожидаемого времени генерации изображений в секундах."""
        model_name = config.diffusion.model.lower()
        num_images = config.diffusion.num_images
        steps = config.diffusion.num_inference_steps
        
        # Базовое время на одно изображение (для одной GPU)
        if "flux" in model_name:
            base_time = 12  # FLUX быстрее
        elif "xl" in model_name:
            base_time = 8   # SDXL
        else:
            base_time = 5   # SD
        
        # Учитываем количество шагов
        time_per_image = base_time * (steps / 20)
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
        Генерация поздравительных картинок с использованием GPU пула.
        
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
            
            # Инициализируем пул если нужно
            if not self.gpu_pool._initialized and config.diffusion.preload_model:
                await self.gpu_pool.initialize()
            
            # Генерируем изображения с использованием GPU пула
            images, content = await self._generate_with_gpu_pool(text)
            
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
                    return output_dir, content
                else:
                    logger.error("❌ Не удалось сохранить ни одного изображения")
                    self._cleanup_directory(output_dir)
                    return None
            else:
                logger.error("❌ Не удалось сгенерировать изображения")
                self._cleanup_directory(output_dir)
                return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации изображений для пользователя {user_id}: {e}")
            if 'output_dir' in locals():
                self._cleanup_directory(output_dir)
            return None

    async def _generate_with_gpu_pool(self, text: str) -> Optional[List[Image.Image]]:
        """
        Генерация изображений с использованием GPU пула.
        
        Args:
            text: Текст поздравления
            
        Returns:
            Список PIL Image или None при ошибке
        """
        try:
            # Создаем промпт
            await self._send_progress_message(
                "translation_start",
                expected_time=self._get_expected_translation_time()
            )
            translation_start_time = time.time()

            prompt, content = await self._create_birthday_prompt(text)
            logger.info(f"📝 Промпт: {prompt}.")
            
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

            # Получаем GPU из пула и генерируем
            async with self.gpu_pool.acquire_gpu() as (device, pipeline):
                logger.info(f"🎮 Генерация на {device}")
                
                # Параметры генерации
                generation_params = self._get_generation_params(prompt)
                
                # Запускаем генерацию в отдельном потоке
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._run_pipeline,
                    pipeline,
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
                    logger.info(f"✅ Успешно сгенерировано {len(images)} изображений на {device}")
                    return images, content
                else:
                    logger.error("❌ Не удалось получить изображения из результата")
                    return None
                
        except Exception as e:
            logger.error(f"❌ Ошибка генерации с GPU пулом: {e}")
            return None

    def _get_generation_params(self, prompt: str) -> dict:
        """Получение параметров генерации."""
        params = {
            "prompt": prompt,
            "height": config.diffusion.height,
            "width": config.diffusion.width,
            "num_inference_steps": config.diffusion.num_inference_steps,
            "guidance_scale": config.diffusion.guidance_scale,
            "num_images_per_prompt": config.diffusion.num_images,
        }
        
        # Добавляем негативный промпт для не-FLUX моделей
        if "flux" not in config.diffusion.model.lower():
            if config.diffusion.negative_prompt:
                params["negative_prompt"] = config.diffusion.negative_prompt
        
        # Добавляем generator для seed
        if config.diffusion.seed >= 0:
            import torch
            params["generator"] = torch.Generator().manual_seed(config.diffusion.seed)
        
        return params

    def _run_pipeline(self, pipeline, params: dict):
        """Запуск pipeline в отдельном потоке."""
        try:
            import torch
            
            with torch.no_grad():
                return pipeline(**params)
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения pipeline: {e}")
            return None

    async def _create_birthday_prompt(self, text: str) -> str:
        """
        Создание промпта для генерации изображения.
        
        Args:
            text: Текст поздравления
            
        Returns:
            Промпт для AI генерации
        """
        prompts_config = config.diffusion.prompts
        
        # Определяем контент на основе языка текста
        if self.has_russian(text):
            content = await self.translate_text(text)
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
        
        return prompt, content

    def has_russian(self, text):
        return bool(re.search(r"[а-яА-ЯёЁ]", text))

    async def translate_text(self, text: str) -> str:
        """
        Перевод текста на английский с использованием пула переводчиков.
        
        Args:
            text: Текст для перевода
            
        Returns:
            str: Переведенный текст
        """
        try:
            # Получаем переводчик из пула
            async with self.translator_pool.acquire_translator() as (device, tokenizer, model):
                logger.debug(f"🔤 Перевод на {device}")
                
                # Запускаем перевод в отдельном потоке
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self._translate_sync,
                    tokenizer,
                    model,
                    text
                )
                
                return result if result else text
                
        except Exception as e:
            logger.error(f"❌ Ошибка перевода текста: {e}")
            return text  # Возвращаем оригинальный текст при ошибке
    
    def _translate_sync(self, tokenizer, model, text: str) -> str:
        """
        Синхронный перевод текста (выполняется в отдельном потоке).
        
        Args:
            tokenizer: Токенайзер модели
            model: Модель перевода
            text: Текст для перевода
            
        Returns:
            str: Переведенный текст
        """
        try:
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # Перемещаем токены на то же устройство что и модель
            if hasattr(model, 'device'):
                tokens = {k: v.to(model.device) for k, v in tokens.items()}
            
            translated = model.generate(**tokens)
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка синхронного перевода: {e}")
            return text

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
        gpu_status = self.gpu_pool.get_status()
        translator_status = self.translator_pool.get_status()
        
        return {
            "local_diffusion_available": self._check_dependencies(),
            "local_model": config.diffusion.model,
            "gpu_pool_status": gpu_status,
            "translator_pool_status": translator_status,
            "num_images_per_generation": config.diffusion.num_images,
            "dependencies_installed": self._check_dependencies()
        }
    