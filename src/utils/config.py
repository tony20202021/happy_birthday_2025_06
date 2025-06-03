"""
Модуль конфигурации для Birthday Bot.
Централизованное управление настройками приложения.
Загружает основную конфигурацию из conf/config.yaml и токены из ../../secrets.yaml
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

@dataclass
class BotConfig:
    """Конфигурация Telegram бота."""
    token: str = ""
    admin_user_id: Optional[int] = None  # ID администратора для получения копий результатов

@dataclass
class SpeechConfig:
    """Конфигурация модуля распознавания речи."""
    model_name: str = "small"
    device: str = "cpu"
    gpu_devices: List[str] = field(default_factory=list)  # Список GPU устройств для multi-GPU Whisper
    language: str = "ru"
    supported_formats: list = field(default_factory=lambda: [".ogg", ".mp3", ".wav", ".m4a", ".flac"])
    temp_audio_dir: str = "temp/audio"
    max_audio_duration: int = 60

@dataclass
class DiffusionPromptsConfig:
    """Конфигурация промптов для генерации изображений."""
    base_picture: str = "cartoon image, fun, joyful, happy"
    style: str = "digital art, professional, masterpiece, best quality"
    subject: str = "young girl named Evelina, dark long hair, big eyes"
    template_with_style: str = "<picture>{picture}</picture>, <style>{style}</style>, <subject>{subject}</subject>, <content>{content}</content>"
    template_no_style: str = "<picture>{picture}</picture>, <subject>{subject}</subject>, <content>{content}</content>"

@dataclass
class DiffusionConfig:
    """Конфигурация локальной генерации изображений."""
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    device: str = "auto"  # auto, cpu, cuda, mps
    gpu_devices: List[str] = field(default_factory=list)  # Список GPU устройств для multi-GPU
    max_queue_size: int = 10  # Максимальный размер очереди ожидания
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    seed: int = -1  # -1 для случайного
    negative_prompt: str = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, signature"
    preload_model: bool = False
    enable_xformers: bool = True
    enable_cpu_offload: bool = True
    num_images: int = 4  # Добавлено поле для количества генерируемых изображений
    prompts: DiffusionPromptsConfig = field(default_factory=DiffusionPromptsConfig)

@dataclass
class PathsConfig:
    """Конфигурация путей к файлам и директориям."""
    temp_dir: str = "temp"
    temp_audio: str = "temp/audio"
    temp_images: str = "temp/images"
    logs_dir: str = "logs"

@dataclass
class SecurityConfig:
    """Настройки безопасности."""
    max_voice_duration: int = 60  # секунды
    rate_limit_messages: int = 10  # сообщений в минуту

@dataclass
class FileConfig:
    """Конфигурация файловых операций."""
    max_file_size: int = 20 * 1024 * 1024  # 20MB
    max_temp_file_age: int = 86400  # 24 часа в секундах

@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(name)s](%(levelname)s) %(filename)s:%(lineno)d: %(message)s"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class DevelopmentConfig:
    """Настройки для разработки."""
    auto_reload: bool = True
    verbose_console: bool = True
    save_debug_images: bool = False

@dataclass
class ProductionConfig:
    """Настройки для продакшена."""
    optimize_memory: bool = True
    cleanup_on_start: bool = True
    performance_monitoring: bool = True

@dataclass
class Config:
    """Основная конфигурация приложения."""
    bot: BotConfig = field(default_factory=BotConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    file: FileConfig = field(default_factory=FileConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)

    def __post_init__(self):
        """Инициализация после создания объекта."""
        self._load_main_config()
        self._load_secrets_config()
        self._load_from_env()
        self._auto_detect_gpus()
        self._create_directories()

    def _auto_detect_gpus(self):
        """Автоопределение доступных GPU устройств."""
        if self.diffusion.device == "auto" and not self.diffusion.gpu_devices:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.diffusion.gpu_devices = [f"cuda:{i}" for i in range(gpu_count)]
                    print(f"✅ Обнаружено {gpu_count} GPU для генерации изображений: {self.diffusion.gpu_devices}")
                else:
                    self.diffusion.gpu_devices = ["cpu"]
                    print("✅ GPU не обнаружены для генерации, используется CPU")
            except ImportError:
                self.diffusion.gpu_devices = ["cpu"]
                print("⚠️ PyTorch не найден для генерации, используется CPU")
        
        # Автоопределение GPU для речи
        if self.speech.device == "auto" and not self.speech.gpu_devices:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    self.speech.gpu_devices = [f"cuda:{i}" for i in range(gpu_count)]
                    print(f"✅ Обнаружено {gpu_count} GPU для распознавания речи: {self.speech.gpu_devices}")
                else:
                    self.speech.gpu_devices = ["cpu"]
                    print("✅ GPU не обнаружены для речи, используется CPU")
            except ImportError:
                self.speech.gpu_devices = ["cpu"]
                print("⚠️ PyTorch не найден для речи, используется CPU")

    def _load_main_config(self):
        """Загрузка основной конфигурации из conf/config.yaml."""
        main_config_path = Path("conf/config.yaml")
        
        if main_config_path.exists():
            try:
                with open(main_config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                if data:
                    self._apply_config_data(data)
                    print(f"✅ Основная конфигурация загружена из {main_config_path}")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки основной конфигурации из {main_config_path}: {e}")
        else:
            print(f"⚠️ Основная конфигурация не найдена: {main_config_path}")

    def _load_secrets_config(self):
        """Загрузка токенов и секретов из ../../secrets.yaml."""
        # Список возможных путей к файлу с секретами
        secrets_paths = [
            "../../secrets.yaml",
            "../../ssh/bot.yaml",
            "../secrets.yaml", 
            "secrets.yaml",
            os.path.expanduser("~/.config/birthday_bot/secrets.yaml"),
            os.path.expanduser("~/.ssh/bot_secrets.yaml"),
        ]
        
        for secrets_path in secrets_paths:
            path = Path(secrets_path)
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    
                    if data:
                        self._apply_config_data(data)
                        print(f"✅ Секреты загружены из {path}")
                        return
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки секретов из {path}: {e}")
        
        print("⚠️ Файл с секретами не найден. Убедитесь, что создан файл ../../secrets.yaml")

    def _apply_config_data(self, data: Dict[str, Any]):
        """Применение данных конфигурации к объекту."""

        # Bot configuration
        if (bot_config := data.get("bot")) is not None:
            if (token := bot_config.get("token")) is not None:
                self.bot.token = token
            if (admin_user_id := bot_config.get("admin_user_id")) is not None:
                self.bot.admin_user_id = int(admin_user_id)

        # Speech configuration
        if (speech_config := data.get("speech")) is not None:
            if (model_name := speech_config.get("model_name")) is not None:
                self.speech.model_name = model_name
            if (device := speech_config.get("device")) is not None:
                self.speech.device = device
            if (gpu_devices := speech_config.get("gpu_devices")) is not None:
                self.speech.gpu_devices = gpu_devices
            if (language := speech_config.get("language")) is not None:
                self.speech.language = language
            if (max_duration := speech_config.get("max_audio_duration")) is not None:
                self.speech.max_audio_duration = max_duration
            if (supported_formats := speech_config.get("supported_formats")) is not None:
                self.speech.supported_formats = supported_formats
            if (temp_dir := speech_config.get("temp_audio_dir")) is not None:
                self.speech.temp_audio_dir = temp_dir

        # Diffusion configuration
        if (diffusion_config := data.get("diffusion")) is not None:
            if (model := diffusion_config.get("model")) is not None:
                self.diffusion.model = model
            if (device := diffusion_config.get("device")) is not None:
                self.diffusion.device = device
            if (gpu_devices := diffusion_config.get("gpu_devices")) is not None:
                self.diffusion.gpu_devices = gpu_devices
            if (max_queue_size := diffusion_config.get("max_queue_size")) is not None:
                self.diffusion.max_queue_size = max_queue_size
            if (width := diffusion_config.get("width")) is not None:
                self.diffusion.width = width
            if (height := diffusion_config.get("height")) is not None:
                self.diffusion.height = height
            if (steps := diffusion_config.get("num_inference_steps")) is not None:
                self.diffusion.num_inference_steps = steps
            if (guidance := diffusion_config.get("guidance_scale")) is not None:
                self.diffusion.guidance_scale = guidance
            if (seed := diffusion_config.get("seed")) is not None:
                self.diffusion.seed = seed
            if (negative := diffusion_config.get("negative_prompt")) is not None:
                self.diffusion.negative_prompt = negative
            if (preload := diffusion_config.get("preload_model")) is not None:
                self.diffusion.preload_model = preload
            if (xformers := diffusion_config.get("enable_xformers")) is not None:
                self.diffusion.enable_xformers = xformers
            if (cpu_offload := diffusion_config.get("enable_cpu_offload")) is not None:
                self.diffusion.enable_cpu_offload = cpu_offload
            if (num_images := diffusion_config.get("num_images")) is not None:
                self.diffusion.num_images = num_images
            
            # Загрузка настроек промптов
            if (prompts_config := diffusion_config.get("prompts")) is not None:
                if (base_picture := prompts_config.get("base_picture")) is not None:
                    self.diffusion.prompts.base_picture = base_picture
                if (style := prompts_config.get("style")) is not None:
                    self.diffusion.prompts.style = style
                if (subject := prompts_config.get("subject")) is not None:
                    self.diffusion.prompts.subject = subject
                if (template_with_style := prompts_config.get("template_with_style")) is not None:
                    self.diffusion.prompts.template_with_style = template_with_style
                if (template_no_style := prompts_config.get("template_no_style")) is not None:
                    self.diffusion.prompts.template_no_style = template_no_style

        # Security configuration
        if (security_config := data.get("security")) is not None:
            if (max_duration := security_config.get("max_voice_duration")) is not None:
                self.security.max_voice_duration = max_duration
            if (rate_limit := security_config.get("rate_limit_messages")) is not None:
                self.security.rate_limit_messages = rate_limit

        # File configuration
        if (file_config := data.get("file")) is not None:
            if (max_size := file_config.get("max_file_size")) is not None:
                self.file.max_file_size = max_size
            if (max_age := file_config.get("max_temp_file_age")) is not None:
                self.file.max_temp_file_age = max_age

        # Paths configuration
        if (paths_config := data.get("paths")) is not None:
            if (temp_dir := paths_config.get("temp_dir")) is not None:
                self.paths.temp_dir = temp_dir
            if (temp_audio := paths_config.get("temp_audio")) is not None:
                self.paths.temp_audio = temp_audio
                self.speech.temp_audio_dir = temp_audio  # Синхронизируем
            if (temp_images := paths_config.get("temp_images")) is not None:
                self.paths.temp_images = temp_images
            if (logs_dir := paths_config.get("logs_dir")) is not None:
                self.paths.logs_dir = logs_dir

        # Logging configuration
        if (logging_config := data.get("logging")) is not None:
            if (level := logging_config.get("level")) is not None:
                self.logging.level = level.upper()
            if (format_str := logging_config.get("format")) is not None:
                self.logging.format = format_str
            if (max_bytes := logging_config.get("max_bytes")) is not None:
                self.logging.max_bytes = max_bytes
            if (backup_count := logging_config.get("backup_count")) is not None:
                self.logging.backup_count = backup_count

        # Development configuration
        if (dev_config := data.get("development")) is not None:
            if (auto_reload := dev_config.get("auto_reload")) is not None:
                self.development.auto_reload = auto_reload
            if (verbose := dev_config.get("verbose_console")) is not None:
                self.development.verbose_console = verbose
            if (debug_images := dev_config.get("save_debug_images")) is not None:
                self.development.save_debug_images = debug_images

        # Production configuration
        if (prod_config := data.get("production")) is not None:
            if (optimize := prod_config.get("optimize_memory")) is not None:
                self.production.optimize_memory = optimize
            if (cleanup := prod_config.get("cleanup_on_start")) is not None:
                self.production.cleanup_on_start = cleanup
            if (monitoring := prod_config.get("performance_monitoring")) is not None:
                self.production.performance_monitoring = monitoring

    def _load_from_env(self):
        """Загрузка конфигурации из переменных окружения (переопределяют файлы)."""
        # Bot configuration
        if token := os.getenv("TELEGRAM_BOT_TOKEN"):
            self.bot.token = token

        # Speech configuration
        if model := os.getenv("WHISPER_MODEL"):
            self.speech.model_name = model
        if device := os.getenv("WHISPER_DEVICE"):
            self.speech.device = device
        if gpu_devices := os.getenv("WHISPER_GPU_DEVICES"):
            self.speech.gpu_devices = [d.strip() for d in gpu_devices.split(",")]
        if language := os.getenv("WHISPER_LANGUAGE"):
            self.speech.language = language

        # Diffusion configuration
        if model := os.getenv("DIFFUSION_MODEL"):
            self.diffusion.model = model
        if device := os.getenv("DIFFUSION_DEVICE"):
            self.diffusion.device = device
        if gpu_devices := os.getenv("DIFFUSION_GPU_DEVICES"):
            self.diffusion.gpu_devices = [d.strip() for d in gpu_devices.split(",")]
        if width := os.getenv("DIFFUSION_WIDTH"):
            try:
                self.diffusion.width = int(width)
            except ValueError:
                pass
        if height := os.getenv("DIFFUSION_HEIGHT"):
            try:
                self.diffusion.height = int(height)
            except ValueError:
                pass
        if num_images := os.getenv("DIFFUSION_NUM_IMAGES"):
            try:
                self.diffusion.num_images = int(num_images)
            except ValueError:
                pass

        # Security configuration
        if max_duration := os.getenv("MAX_VOICE_DURATION"):
            try:
                self.security.max_voice_duration = int(max_duration)
                self.speech.max_audio_duration = int(max_duration)
            except ValueError:
                pass

        # Logging configuration
        if log_level := os.getenv("LOG_LEVEL"):
            self.logging.level = log_level.upper()

    def _create_directories(self):
        """Создание необходимых директорий."""
        directories = [
            self.paths.temp_dir,
            self.paths.temp_audio,
            self.paths.temp_images,
            self.paths.logs_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def get_temp_audio_path(self, filename: str) -> str:
        """Получение полного пути к временному аудиофайлу."""
        return str(Path(self.paths.temp_audio) / filename)

    def get_temp_image_path(self, filename: str) -> str:
        """Получение полного пути к временному изображению."""
        return str(Path(self.paths.temp_images) / filename)

    def get_temp_images_dir(self, user_id: int) -> str:
        """
        Получение пути к временной директории для изображений конкретного пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Путь к директории для изображений пользователя
        """
        timestamp = int(time.time())
        dir_name = f"birthday_cards_{user_id}_{timestamp}"
        return str(Path(self.paths.temp_images) / dir_name)

    def create_temp_images_dir(self, user_id: int) -> str:
        """
        Создание временной директории для изображений пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Путь к созданной директории
        """
        temp_dir = self.get_temp_images_dir(user_id)
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        return temp_dir

    def validate(self) -> list:
        """
        Валидация конфигурации.
        
        Returns:
            Список ошибок конфигурации
        """
        errors = []

        # Проверка токена бота
        if not self.bot.token or self.bot.token == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
            errors.append("❌ Не указан токен Telegram бота. Создайте файл ../../secrets.yaml с корректным токеном")

        # Проверка модели Whisper
        valid_whisper_models = ["tiny", "base", "small", "medium", "large"]
        if self.speech.model_name not in valid_whisper_models:
            errors.append(f"❌ Неверная модель Whisper: {self.speech.model_name}. "
                         f"Доступные: {', '.join(valid_whisper_models)}")

        # Проверка устройства для генерации изображений
        if self.diffusion.device not in ["cpu", "cuda", "mps", "auto"]:
            errors.append(f"❌ Неверное устройство для генерации: {self.diffusion.device}. "
                         f"Доступные: cpu, cuda, mps, auto")

        # Проверка устройства Whisper
        if self.speech.device not in ["cpu", "cuda", "auto"]:
            errors.append(f"❌ Неверное устройство Whisper: {self.speech.device}. "
                         f"Доступные: cpu, cuda, auto")

        # Проверка количества изображений
        if not (1 <= self.diffusion.num_images <= 10):
            errors.append(f"❌ Неверное количество изображений: {self.diffusion.num_images}. "
                         f"Должно быть от 1 до 10")

        # Проверка GPU устройств
        if self.diffusion.gpu_devices:
            for device in self.diffusion.gpu_devices:
                if not (device == "cpu" or device.startswith("cuda:") or device == "mps"):
                    errors.append(f"❌ Неверное GPU устройство для генерации: {device}")
        
        if self.speech.gpu_devices:
            for device in self.speech.gpu_devices:
                if not (device == "cpu" or device.startswith("cuda:") or device == "mps"):
                    errors.append(f"❌ Неверное GPU устройство для речи: {device}")

        # Проверка PyTorch для diffusion и speech
        try:
            import torch
            for device in self.diffusion.gpu_devices:
                if device.startswith("cuda") and not torch.cuda.is_available():
                    errors.append("❌ CUDA недоступна, но указана в diffusion.gpu_devices")
                elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    errors.append("❌ MPS недоступна, но указана в diffusion.gpu_devices")
            
            for device in self.speech.gpu_devices:
                if device.startswith("cuda") and not torch.cuda.is_available():
                    errors.append("❌ CUDA недоступна, но указана в speech.gpu_devices")
                elif device == "mps" and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                    errors.append("❌ MPS недоступна, но указана в speech.gpu_devices")
        except ImportError:
            errors.append("❌ PyTorch не установлен (требуется для локальной генерации изображений)")

        # Проверка директорий
        for attr_name in ["temp_dir", "temp_audio", "temp_images", "logs_dir"]:
            path = getattr(self.paths, attr_name)
            if not Path(path).exists():
                errors.append(f"❌ Директория не существует: {path}")

        return errors

    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса конфигурации.
        
        Returns:
            Словарь с информацией о статусе
        """
        return {
            "bot_token_configured": bool(self.bot.token and self.bot.token != "YOUR_TELEGRAM_BOT_TOKEN_HERE"),
            "bot_admin_user_id": self.bot.admin_user_id,
            "diffusion_model": self.diffusion.model,
            "diffusion_device": self.diffusion.device,
            "diffusion_gpu_devices": self.diffusion.gpu_devices,
            "diffusion_num_images": self.diffusion.num_images,
            "speech_model": self.speech.model_name,
            "speech_device": self.speech.device,
            "speech_gpu_devices": self.speech.gpu_devices,
            "temp_directories_exist": all(
                Path(getattr(self.paths, attr)).exists()
                for attr in ["temp_dir", "temp_audio", "temp_images", "logs_dir"]
            ),
            "max_voice_duration": self.security.max_voice_duration,
            "rate_limit": self.security.rate_limit_messages,
            "torch_available": self._check_torch(),
            "diffusers_available": self._check_diffusers(),
            "whisper_available": self._check_whisper(),
        }

    def _check_torch(self) -> bool:
        """Проверка доступности PyTorch."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def _check_diffusers(self) -> bool:
        """Проверка доступности diffusers."""
        try:
            import diffusers
            return True
        except ImportError:
            return False

    def _check_whisper(self) -> bool:
        """Проверка доступности Whisper."""
        try:
            import whisper
            return True
        except ImportError:
            return False

    def is_development_mode(self) -> bool:
        """Проверка режима разработки."""
        return (self.logging.level == "DEBUG" or 
                os.getenv("ENVIRONMENT", "").lower() in ["dev", "development"])

    def is_production_mode(self) -> bool:
        """Проверка продакшен режима."""
        return (os.getenv("ENVIRONMENT", "").lower() in ["prod", "production"] or
                not self.is_development_mode())

# Создаем глобальный экземпляр конфигурации
config = Config()

# Сообщения бота (теперь могут использовать конфигурацию)
BOT_MESSAGES = {
    "start": (
        "🎉 <b>Добро пожаловать в Birthday Bot!</b>\n\n"
        "Я умею создавать красивые поздравительные картинки с днем рождения! 🎂\n\n"
        "📝 <b>Как пользоваться:</b>\n"
        "• Отправьте мне текст поздравления\n"
        "• Или запишите голосовое сообщение\n"
        f"• Я создам для вас {config.diffusion.num_images} красивые картинки! 🎨\n\n"
        "🎤 <i>Голосовые сообщения распознаются с помощью OpenAI Whisper</i>\n"
        "🖼️ <i>Изображения генерируются локально с помощью Stable Diffusion</i>\n\n"
        "Попробуйте прямо сейчас! 🚀"
    ),
    "help": (
        "ℹ️ <b>Справка по Birthday Bot</b>\n\n"
        "🎯 <b>Возможности:</b>\n"
        "• Создание поздравительных картинок с AI\n"
        "• Распознавание голосовых сообщений\n"
        "• Локальная генерация изображений\n\n"
        "📝 <b>Поддерживаемые форматы:</b>\n"
        "• Текстовые сообщения любой длины\n"
        f"• Голосовые сообщения (до {config.security.max_voice_duration} секунд)\n\n"
        "🎨 <b>Технологии:</b>\n"
        f"• OpenAI Whisper ({config.speech.model_name}) для распознавания речи\n"
        f"• Stable Diffusion ({config.diffusion.model.split('/')[-1]}) для генерации изображений\n"
        f"• Генерируется {config.diffusion.num_images} изображения за раз\n\n"
        "❓ <b>Проблемы?</b> Попробуйте команду /start"
    ),
    "processing": "⏳ Обрабатываю ваше сообщение...",
    "processing_voice": "🎤 Распознаю голосовое сообщение...",
    "generating_image": f"🎨 Создаю {config.diffusion.num_images} поздравительные картинки...",
    "voice_too_long": (
        "⚠️ Голосовое сообщение слишком длинное!\n"
        f"Максимальная длительность: {config.security.max_voice_duration} секунд"
    ),
    "error": (
        "❌ Произошла ошибка при обработке вашего запроса.\n"
        "Попробуйте еще раз или обратитесь к администратору."
    ),
    
    # Сообщения о прогрессе для долгих операций
    "progress": {
        "speech_recognition_start": "🎤 <b>Распознавание речи</b>\n⏱️ Ожидаемое время: {expected_time} сек",
        "speech_recognition_done": "✅ <b>Речь распознана</b>\n⏱️ Время выполнения: {actual_time:.1f} сек",
        
        "model_loading_start": "📥 <b>Загрузка AI модели</b>\n⏱️ Ожидаемое время: {expected_time} сек",
        "model_loading_done": "✅ <b>Модель загружена</b>\n⏱️ Время выполнения: {actual_time:.1f} сек",
        
        "image_generation_start": "🎨 <b>Генерация {num_images} изображений</b>\n⏱️ Ожидаемое время: {expected_time} сек",
        "image_generation_done": "✅ <b>Изображения сгенерированы</b>\n⏱️ Время выполнения: {actual_time:.1f} сек",
        
        "translation_start": "🎨 <b>Перевод текста</b>\n⏱️ Ожидаемое время: {expected_time} сек",
        "translation_done": "✅ <b>Текст переведен</b>\n⏱️ Время выполнения: {actual_time:.1f} сек",
        
        "sending_images": "📤 <b>Отправка изображений...</b>"
    }
}
