"""
Модуль обработчиков сообщений для Birthday Bot.
Содержит обработчики команд и различных типов сообщений.
"""

import asyncio
import time
import shutil
from pathlib import Path
from typing import Any, List

from aiogram import Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, Voice, InputMediaPhoto
from aiogram.types import FSInputFile
from aiogram.fsm.context import FSMContext

from src.utils.config import config, BOT_MESSAGES
from src.utils.logger import get_handlers_logger
from src.speech.speech_to_text import SpeechToText
from src.image.generator import ImageGenerator

# Инициализация логгера
logger = get_handlers_logger()

# Инициализация модулей (будут созданы при первом использовании)
speech_processor = None
image_generator = None

def get_speech_processor():
    """Получение экземпляра обработчика речи (lazy initialization)."""
    global speech_processor
    if speech_processor is None:
        try:
            speech_processor = SpeechToText()
            logger.info("✅ Модуль распознавания речи инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации модуля речи: {e}")
            speech_processor = None
    return speech_processor

def get_image_generator():
    """Получение экземпляра генератора изображений (lazy initialization)."""
    global image_generator
    if image_generator is None:
        try:
            # Создаем генератор без callback - он будет установлен позже
            image_generator = ImageGenerator()
            logger.info("✅ Модуль генерации изображений инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации генератора изображений: {e}")
            image_generator = None
    return image_generator

async def send_media_group_from_directory(message: Message, images_dir: str) -> bool:
    """
    Отправка медиа-группы с изображениями из директории.
    
    Args:
        message: Сообщение пользователя для ответа
        images_dir: Путь к директории с изображениями
        
    Returns:
        bool: True если изображения отправлены успешно
    """
    try:
        generator = get_image_generator()
        if not generator:
            logger.error("❌ Генератор изображений недоступен")
            return False
            
        # Получаем пути ко всем изображениям
        image_paths = generator.get_image_paths_from_dir(images_dir)
        
        if not image_paths:
            logger.error(f"❌ Не найдено изображений в директории: {images_dir}")
            return False
        
        logger.info(f"📤 Отправка {len(image_paths)} изображений")
        
        # Если одно изображение - отправляем как обычное фото
        if len(image_paths) == 1:
            photo = FSInputFile(image_paths[0])
            await message.answer_photo(
                photo=photo,
                caption="🎉 Ваша поздравительная картинка готова!"
            )
            logger.info("✅ Отправлено одно изображение")
            return True
        
        # Если несколько изображений - отправляем как медиа-группу
        media_group = []
        for i, image_path in enumerate(image_paths):
            # Первое изображение с подписью, остальные без
            caption = "🎉 Ваши поздравительные картинки готовы!" if i == 0 else None
            
            media_group.append(
                InputMediaPhoto(
                    media=FSInputFile(image_path),
                    caption=caption
                )
            )
        
        # Отправляем медиа-группу
        await message.answer_media_group(media=media_group)
        logger.info(f"✅ Отправлена медиа-группа из {len(image_paths)} изображений")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка отправки медиа-группы: {e}")
        return False

async def create_progress_callback(message: Message):
    """
    Создание callback функции для отправки сообщений о прогрессе.
    
    Args:
        message: Сообщение пользователя для ответа
        
    Returns:
        Функция callback для отправки прогресса
    """
    progress_messages = {}  # Словарь для хранения ID сообщений о прогрессе
    
    async def progress_callback(message_key: str, **kwargs):
        try:
            progress_text = BOT_MESSAGES["progress"][message_key].format(**kwargs)
            
            # Если это начало этапа - отправляем новое сообщение
            if message_key.endswith("_start"):
                progress_msg = await message.answer(progress_text, parse_mode="HTML")
                progress_messages[message_key] = progress_msg.message_id
            
            # Если это завершение этапа - редактируем существующее сообщение
            elif message_key.endswith("_done"):
                start_key = message_key.replace("_done", "_start")
                if start_key in progress_messages:
                    try:
                        await message.bot.answer(
                            text=progress_text,
                            # chat_id=message.chat.id,
                            # message_id=progress_messages[start_key],
                            parse_mode="HTML"
                        )
                    except Exception as e:
                        logger.debug(f"Не удалось отредактировать сообщение: {e}")
                        # Если не удалось отредактировать, отправляем новое
                        await message.answer(progress_text, parse_mode="HTML")
            
            # Для других типов сообщений
            else:
                await message.answer(progress_text, parse_mode="HTML")
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка отправки сообщения о прогрессе: {e}")
    
    return progress_callback

def get_expected_speech_time(duration_seconds: int) -> int:
    """Получение ожидаемого времени распознавания речи."""
    model_name = config.speech.model_name
    device = config.speech.device
    
    # Базовое время обработки в зависимости от модели и устройства
    if device == "cuda":
        base_multiplier = {
            "tiny": 0.01,
            "base": 0.05, 
            "small": 0.1,
            "medium": 0.2,
            "large": 0.3
        }
    else:  # CPU
        base_multiplier = {
            "tiny": 0.1,
            "base": 0.2,
            "small": 0.3,
            "medium": 0.7,
            "large": 1.0
        }
    
    multiplier = base_multiplier.get(model_name, 1.0)
    expected_time = int((duration_seconds + 1) * multiplier)  # +5 сек на загрузку
    
    return max(expected_time, 3)  # Минимум 3 секунды

def cleanup_images_directory(images_dir: str) -> None:
    """
    Очистка директории с изображениями.
    
    Args:
        images_dir: Путь к директории для удаления
    """
    try:
        dir_path = Path(images_dir)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            logger.debug(f"🗑️ Удалена директория: {images_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось удалить директорию {images_dir}: {e}")
    """
    Очистка директории с изображениями.
    
    Args:
        images_dir: Путь к директории для удаления
    """
    try:
        dir_path = Path(images_dir)
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            logger.debug(f"🗑️ Удалена директория: {images_dir}")
    except Exception as e:
        logger.warning(f"⚠️ Не удалось удалить директорию {images_dir}: {e}")

async def cmd_start(message: Message, state: FSMContext):
    """
    Обработчик команды /start.
    
    Args:
        message: Сообщение пользователя
        state: Состояние FSM
    """
    try:
        # Очищаем состояние пользователя
        await state.clear()
        
        # Логируем взаимодействие
        user_info = f"{message.from_user.full_name} (@{message.from_user.username or 'unknown'})"
        logger.info(f"👤 Пользователь {user_info} выполнил START_COMMAND")
        
        # Отправляем приветственное сообщение
        await message.answer(
            BOT_MESSAGES["start"],
            parse_mode="HTML"
        )
        
        logger.info(f"✅ Отправлено приветствие пользователю {message.from_user.full_name}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в cmd_start для пользователя {message.from_user.full_name}: {e}")
        await message.answer(BOT_MESSAGES["error"])

async def cmd_help(message: Message):
    """
    Обработчик команды /help.
    
    Args:
        message: Сообщение пользователя
    """
    try:
        # Логируем взаимодействие
        user_info = f"{message.from_user.full_name} (@{message.from_user.username or 'unknown'})"
        logger.info(f"👤 Пользователь {user_info} выполнил HELP_COMMAND")
        
        # Отправляем справочное сообщение
        await message.answer(
            BOT_MESSAGES["help"],
            parse_mode="HTML"
        )
        
        logger.info(f"✅ Отправлена справка пользователю {message.from_user.full_name}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка в cmd_help для пользователя {message.from_user.full_name}: {e}")
        await message.answer(BOT_MESSAGES["error"])

async def handle_text_message(message: Message):
    """
    Обработчик текстовых сообщений.
    
    Args:
        message: Текстовое сообщение пользователя
    """
    start_time = time.time()
    
    try:
        # Логируем получение текстового сообщения
        text_preview = message.text[:50] + "..." if len(message.text) > 50 else message.text
        logger.info(f"📝 Пользователь {message.from_user.full_name} отправил TEXT_MESSAGE длиной {len(message.text)} символов")
        logger.debug(f"   Текст: {text_preview}")
        
        # Отправляем сообщение о начале обработки
        progress_callback = await create_progress_callback(message)
        
        # Генерируем изображения
        generator = get_image_generator()
        if not generator:
            await message.answer("❌ Сервис генерации изображений временно недоступен. Попробуйте позже.")
            logger.error(f"❌ Генератор изображений недоступен для пользователя {message.from_user.full_name}")
            return
        
        # Устанавливаем callback для прогресса
        generator.progress_callback = progress_callback
        
        # Генерируем изображения (возвращается путь к директории)
        images_dir = await generator.generate_birthday_image(message.text, message.from_user.id)
        
        if images_dir and Path(images_dir).exists():
            # Отправляем сообщение об отправке
            await progress_callback("sending_images")
            
            # Отправляем сгенерированные изображения
            success = await send_media_group_from_directory(message, images_dir)            
            if success:
                processing_time = time.time() - start_time
                logger.info(f"✅ Изображения успешно сгенерированы и отправлены пользователю {message.from_user.full_name} за {processing_time:.2f}с")
                
                # Логируем статистику генерации изображения
                logger.info(
                    f"user_id={message.from_user.id},"
                    f"prompt_length={len(message.text)},"
                    f"generation_time={processing_time},"
                    f"num_images={config.diffusion.num_images},"
                    f"success={True}"
                )
            else:
                await message.answer("❌ Не удалось отправить изображения. Попробуйте еще раз.")
                logger.error(f"❌ Не удалось отправить изображения пользователю {message.from_user.full_name}")
                
                # Логируем неудачную отправку
                processing_time = time.time() - start_time
                logger.info(
                    f"user_id={message.from_user.id},"
                    f"prompt_length={len(message.text)},"
                    f"generation_time={processing_time},"
                    f"num_images={config.diffusion.num_images},"
                    f"success={False}"
                )
            
            # Удаляем временную директорию с изображениями
            cleanup_images_directory(images_dir)
            
        else:
            await message.answer("❌ Не удалось создать изображения. Попробуйте еще раз.")
            logger.error(f"❌ Не удалось создать изображения для пользователя {message.from_user.full_name}")
            
            # Логируем неудачную генерацию
            processing_time = time.time() - start_time
            logger.info(
                f"user_id={message.from_user.id},"
                f"prompt_length={len(message.text)},"
                f"generation_time={processing_time},"
                f"num_images={config.diffusion.num_images},"
                f"success={False}"
            )
        
        # Логируем общее время обработки
        total_time = time.time() - start_time
        logger.info(
            f"user_id={message.from_user.id},"
            f"message_type=text,"
            f"processing_time={total_time}"
        )
        
    except Exception as e:
        logger.error(
            f"error={e},"
            f"context={{"
            f'"method": "handle_text_message",'
            f'"user_id": {message.from_user.id},'
            f'"text_length": {len(message.text) if message.text else 0}'
            f"}}"
        )
        await message.answer(BOT_MESSAGES["error"])

async def handle_voice_message(message: Message):
    """
    Обработчик голосовых сообщений.
    
    Args:
        message: Голосовое сообщение пользователя
    """
    start_time = time.time()
    
    try:
        voice: Voice = message.voice
        
        # Проверяем длительность голосового сообщения
        if voice.duration > config.security.max_voice_duration:
            await message.answer(BOT_MESSAGES["voice_too_long"])
            logger.warning(f"⚠️ Пользователь {message.from_user.full_name} отправил слишком длинное голосовое сообщение ({voice.duration}с)")
            return
        
        # Логируем получение голосового сообщения
        logger.info(f"🎤 Пользователь {message.from_user.full_name} отправил VOICE_MESSAGE длительностью {voice.duration}с, размером {voice.file_size} байт")
        
        # Отправляем сообщение о начале обработки
        progress_callback = await create_progress_callback(message)
        
        # Получаем процессор речи
        speech_processor = get_speech_processor()
        if not speech_processor:
            await message.answer("❌ Сервис распознавания речи временно недоступен. Попробуйте позже.")
            logger.error(f"❌ Процессор речи недоступен для пользователя {message.from_user.full_name}")
            return
        
        # Отправляем сообщение о начале распознавания речи
        await progress_callback(
            "speech_recognition_start",
            expected_time=get_expected_speech_time(voice.duration)
        )
        
        # Распознаем речь через специальный метод для Telegram
        speech_start_time = time.time()
        recognized_text = await speech_processor.transcribe_telegram_voice(
            bot=message.bot,
            voice_message=voice,
            user_id=message.from_user.id
        )
        speech_time = time.time() - speech_start_time
        
        # Отправляем сообщение о завершении распознавания
        await progress_callback(
            "speech_recognition_done",
            actual_time=speech_time
        )
        
        if recognized_text:
            # Логируем успешное распознавание
            logger.info(f"✅ Речь пользователя {message.from_user.full_name} успешно распознана за {speech_time:.2f}с")
            logger.debug(f"   Распознанный текст: {recognized_text}")
            
            # Отправляем распознанный текст
            await message.answer(
                f"🎤 Распознанный текст:\n<i>{recognized_text}</i>",
                parse_mode="HTML"
            )
            
            # Генерируем изображения на основе распознанного текста
            generator = get_image_generator()
            if not generator:
                await message.answer("❌ Сервис генерации изображений временно недоступен.")
                logger.error(f"❌ Генератор изображений недоступен для пользователя {message.from_user.full_name}")
                return
            
            # Устанавливаем callback для прогресса
            generator.progress_callback = progress_callback
            
            # Генерируем изображения (возвращается путь к директории)
            images_dir = await generator.generate_birthday_image(recognized_text, message.from_user.id)
            
            if images_dir and Path(images_dir).exists():
                # Отправляем сообщение об отправке
                await progress_callback("sending_images")
                
                # Отправляем сгенерированные изображения
                success = await send_media_group_from_directory(message, images_dir)
                
                if success:
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Изображения по голосовому сообщению успешно созданы для пользователя {message.from_user.full_name} за {processing_time:.2f}с")
                    
                    # Логируем статистику генерации изображения
                    logger.info(
                        f"user_id={message.from_user.id},"
                        f"prompt_length={len(recognized_text)},"
                        f"generation_time={processing_time - speech_time},"
                        f"num_images={config.diffusion.num_images},"
                        f"success={True}"
                    )
                else:
                    await message.answer("❌ Не удалось отправить изображения. Попробуйте еще раз.")
                    logger.error(f"❌ Не удалось отправить изображения по голосовому сообщению для пользователя {message.from_user.full_name}")
                    
                    # Логируем неудачную отправку
                    processing_time = time.time() - start_time
                    logger.info(
                        f"user_id={message.from_user.id},"
                        f"prompt_length={len(recognized_text)},"
                        f"generation_time={processing_time - speech_time},"
                        f"num_images={config.diffusion.num_images},"
                        f"success={False}"
                    )
                
                # Удаляем временную директорию с изображениями
                cleanup_images_directory(images_dir)
                
            else:
                await message.answer("❌ Не удалось создать изображения. Попробуйте еще раз.")
                logger.error(f"❌ Не удалось создать изображения по голосовому сообщению для пользователя {message.from_user.full_name}")
                
                # Логируем неудачную генерацию
                processing_time = time.time() - start_time
                logger.info(
                    f"user_id={message.from_user.id},"
                    f"prompt_length={len(recognized_text)},"
                    f"generation_time={processing_time - speech_time},"
                    f"num_images={config.diffusion.num_images},"
                    f"success={False}"
                )
        else:
            # Логируем неуспешное распознавание
            logger.warning(f"⚠️ Не удалось распознать речь пользователя {message.from_user.full_name}")
            await message.answer("❌ Не удалось распознать речь. Попробуйте говорить четче.")
        
        # Логируем общее время обработки
        total_time = time.time() - start_time
        logger.info(
            f"user_id={message.from_user.id},"
            f"message_type=voice,"
            f"processing_time={total_time}"
        )
        
    except Exception as e:
        logger.error(
            f"error={e},"
            f"context={{"
            f'"method": "handle_voice_message",'
            f'"user_id": {message.from_user.id},'
            f'"voice_duration": {getattr(voice, "duration", "unknown") if "voice" in locals() else "unknown"}'
            f"}}"
        )
        await message.answer(BOT_MESSAGES["error"])

async def handle_unsupported_content(message: Message):
    """
    Обработчик неподдерживаемого контента.
    
    Args:
        message: Сообщение с неподдерживаемым контентом
    """
    try:
        # Определяем тип контента
        content_type = "unknown"
        if message.photo:
            content_type = "photo"
        elif message.video:
            content_type = "video"
        elif message.document:
            content_type = "document"
        elif message.sticker:
            content_type = "sticker"
        elif message.animation:
            content_type = "animation"
        elif message.video_note:
            content_type = "video_note"
        elif message.location:
            content_type = "location"
        elif message.contact:
            content_type = "contact"
        
        # Логируем получение неподдерживаемого контента
        logger.info(f"❓ Пользователь {message.from_user.full_name} отправил UNSUPPORTED_CONTENT типа: {content_type}")
        
        await message.answer(
            "🤔 Я пока умею работать только с текстовыми и голосовыми сообщениями.\n\n"
            f"📝 Отправьте мне текст с поздравлением или запишите голосовое сообщение!\n"
            f"🎨 Я создам для вас {config.diffusion.num_images} красивые картинки!"
        )
        
    except Exception as e:
        logger.error(
            f"error={e},"
            f"context={{"
            f'"method": "handle_unsupported_content",'
            f'"user_id": {message.from_user.id}'
            f"}}"
        )
        await message.answer(BOT_MESSAGES["error"])

def register_handlers(dp: Dispatcher) -> None:
    """
    Регистрация всех обработчиков сообщений.
    
    Args:
        dp: Диспетчер aiogram
    """
    logger.info("🔧 Регистрация обработчиков сообщений...")
    
    try:
        # Регистрация обработчиков команд
        dp.message.register(cmd_start, Command("start"))
        dp.message.register(cmd_help, Command("help"))
        
        # Регистрация обработчиков контента
        dp.message.register(handle_voice_message, F.voice)
        dp.message.register(handle_text_message, F.text)
        
        # Обработчик неподдерживаемого контента (должен быть последним)
        dp.message.register(
            handle_unsupported_content,
            F.photo | F.video | F.document | F.sticker | F.animation | 
            F.video_note | F.location | F.contact
        )
        
        logger.info("✅ Все обработчики успешно зарегистрированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка регистрации обработчиков: {e}")
        raise

# Функции для работы с rate limiting (заглушки для будущей реализации)
async def check_rate_limit(user_id: int) -> bool:
    """
    Проверка лимита сообщений для пользователя.
    
    Args:
        user_id: ID пользователя
        
    Returns:
        bool: True если пользователь не превысил лимит
    """
    # TODO: Реализовать rate limiting с использованием Redis или in-memory cache
    return True

async def update_rate_limit(user_id: int) -> None:
    """
    Обновление счетчика сообщений для пользователя.
    
    Args:
        user_id: ID пользователя
    """
    # TODO: Реализовать обновление rate limiting
    pass
