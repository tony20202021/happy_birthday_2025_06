"""
Модуль менеджера Telegram бота.
Отвечает за управление экземпляром бота и диспетчера.
"""

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, ChatMember
from aiogram.types import FSInputFile
from typing import List, Union, Dict, Any
from pathlib import Path

from src.utils.logger import get_bot_logger

# Команды бота
BOT_COMMANDS = [
    {"command": "start", "description": "🎉 Начать работу с ботом"},
    {"command": "help", "description": "❓ Получить справку"},
]

class BotManager:
    """Менеджер бота для управления экземпляром бота и диспетчера."""
    
    def __init__(self, bot: Bot, dp: Dispatcher):
        """
        Инициализация менеджера бота.
        
        Args:
            bot: Экземпляр бота
            dp: Экземпляр диспетчера
        """
        self.bot = bot
        self.dp = dp
        self.logger = get_bot_logger()
        
        # Информация о боте (будет заполнена при запуске)
        self.bot_info = None
        
        self.logger.info("🤖 Инициализирован менеджер бота")
        
    async def setup_commands(self) -> None:
        """
        Настройка команд бота для отображения в меню Telegram.
        """
        try:
            commands = [
                BotCommand(command=cmd["command"], description=cmd["description"]) 
                for cmd in BOT_COMMANDS
            ]
            
            await self.bot.set_my_commands(commands)
            self.logger.info(f"✅ Настроено {len(commands)} команд бота")
            
            # Логируем установленные команды
            for cmd in commands:
                self.logger.debug(f"   Command: /{cmd.command} - {cmd.description}")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка настройки команд бота: {e}")
            raise
    
    async def get_bot_info(self) -> Dict[str, Any]:
        """
        Получение информации о боте.
        
        Returns:
            dict: Информация о боте
        """
        try:
            if self.bot_info is None:
                me = await self.bot.get_me()
                self.bot_info = {
                    "id": me.id,
                    "username": me.username,
                    "first_name": me.first_name,
                    "is_bot": me.is_bot,
                    "can_join_groups": me.can_join_groups,
                    "can_read_all_group_messages": me.can_read_all_group_messages,
                    "supports_inline_queries": me.supports_inline_queries
                }
                
                self.logger.info(f"📋 Информация о боте получена: @{me.username}")
                self.logger.debug(f"   ID: {me.id}")
                self.logger.debug(f"   Имя: {me.first_name}")
                self.logger.debug(f"   Может работать в группах: {me.can_join_groups}")
            
            return self.bot_info
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о боте: {e}")
            return {}
    
    async def send_message_safe(self, chat_id: int, text: str, **kwargs) -> bool:
        """
        Безопасная отправка сообщения с обработкой ошибок.
        
        Args:
            chat_id: ID чата
            text: Текст сообщения
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если сообщение отправлено успешно
        """
        try:
            await self.bot.send_message(chat_id=chat_id, text=text, **kwargs)
            self.logger.debug(f"✅ Сообщение отправлено в чат {chat_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки сообщения в чат {chat_id}: {e}")
            return False
    
    async def send_photo_safe(self, chat_id: int, photo: Union[str, Path, FSInputFile], 
                             caption: str = None, **kwargs) -> bool:
        """
        Безопасная отправка фото с обработкой ошибок.
        
        Args:
            chat_id: ID чата
            photo: Фото для отправки (путь к файлу, Path или FSInputFile)
            caption: Подпись к фото
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если фото отправлено успешно
        """
        try:
            # Если передан путь к файлу, конвертируем в FSInputFile
            if isinstance(photo, (str, Path)):
                photo_path = Path(photo)
                if not photo_path.exists():
                    self.logger.error(f"❌ Файл фото не найден: {photo_path}")
                    return False
                photo = FSInputFile(photo_path)
            
            await self.bot.send_photo(chat_id=chat_id, photo=photo, caption=caption, **kwargs)
            self.logger.debug(f"✅ Фото отправлено в чат {chat_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки фото в чат {chat_id}: {e}")
            return False
    
    async def download_file_safe(self, file_id: str, destination: Union[str, Path]) -> bool:
        """
        Безопасное скачивание файла с обработкой ошибок.
        
        Args:
            file_id: ID файла для скачивания
            destination: Путь для сохранения файла
            
        Returns:
            bool: True если файл скачан успешно
        """
        try:
            file = await self.bot.get_file(file_id)
            await self.bot.download_file(file.file_path, destination)
            self.logger.debug(f"✅ Файл {file_id} скачан в {destination}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка скачивания файла {file_id}: {e}")
            return False
    
    def get_user_info_string(self, user) -> str:
        """
        Получение строкового представления информации о пользователе.
        
        Args:
            user: Объект пользователя Telegram
            
        Returns:
            str: Строковое представление пользователя
        """
        username = f"@{user.username}" if user.username else "без username"
        full_name = f"{user.first_name}"
        if user.last_name:
            full_name += f" {user.last_name}"
        
        return f"{full_name} ({username}, ID: {user.id})"
    
    async def log_user_interaction(self, user, message_type: str, content_preview: str = ""):
        """
        Логирование взаимодействия с пользователем.
        
        Args:
            user: Объект пользователя Telegram
            message_type: Тип сообщения
            content_preview: Превью содержимого
        """
        user_info = self.get_user_info_string(user)
        
        log_message = f"USER_INTERACTION | {user_info} | TYPE: {message_type}"
        if content_preview:
            # Обрезаем превью до 100 символов
            preview = content_preview[:100] + "..." if len(content_preview) > 100 else content_preview
            log_message += f" | PREVIEW: {preview}"
        
        self.logger.info(log_message)
    
    async def get_chat_member_count(self, chat_id: int) -> int:
        """
        Получение количества участников чата.
        
        Args:
            chat_id: ID чата
            
        Returns:
            int: Количество участников или -1 при ошибке
        """
        try:
            count = await self.bot.get_chat_member_count(chat_id)
            self.logger.debug(f"Количество участников в чате {chat_id}: {count}")
            return count
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения количества участников чата {chat_id}: {e}")
            return -1
    
    async def check_bot_permissions(self, chat_id: int) -> Dict[str, Any]:
        """
        Проверка разрешений бота в чате.
        
        Args:
            chat_id: ID чата
            
        Returns:
            dict: Словарь с разрешениями бота
        """
        try:
            me = await self.bot.get_me()
            chat_member: ChatMember = await self.bot.get_chat_member(chat_id, me.id)
            
            permissions = {
                "status": chat_member.status,
                "can_send_messages": getattr(chat_member, "can_send_messages", True),
                "can_send_media_messages": getattr(chat_member, "can_send_media_messages", True),
                "can_send_other_messages": getattr(chat_member, "can_send_other_messages", True),
                "can_add_web_page_previews": getattr(chat_member, "can_add_web_page_previews", True),
            }
            
            self.logger.debug(f"Разрешения бота в чате {chat_id}: {permissions}")
            return permissions
        except Exception as e:
            self.logger.error(f"❌ Ошибка проверки разрешений в чате {chat_id}: {e}")
            return {"status": "unknown", "error": str(e)}
    
    async def get_chat_info(self, chat_id: int) -> Dict[str, Any]:
        """
        Получение информации о чате.
        
        Args:
            chat_id: ID чата
            
        Returns:
            dict: Информация о чате
        """
        try:
            chat = await self.bot.get_chat(chat_id)
            
            chat_info = {
                "id": chat.id,
                "type": chat.type,
                "title": getattr(chat, "title", None),
                "username": getattr(chat, "username", None),
                "first_name": getattr(chat, "first_name", None),
                "last_name": getattr(chat, "last_name", None),
                "description": getattr(chat, "description", None),
            }
            
            self.logger.debug(f"Информация о чате {chat_id}: {chat_info}")
            return chat_info
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о чате {chat_id}: {e}")
            return {"id": chat_id, "error": str(e)}
    
    async def send_typing_action(self, chat_id: int) -> bool:
        """
        Отправка индикатора печати.
        
        Args:
            chat_id: ID чата
            
        Returns:
            bool: True если индикатор отправлен успешно
        """
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action="typing")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки индикатора печати в чат {chat_id}: {e}")
            return False
    
    async def send_upload_photo_action(self, chat_id: int) -> bool:
        """
        Отправка индикатора загрузки фото.
        
        Args:
            chat_id: ID чата
            
        Returns:
            bool: True если индикатор отправлен успешно
        """
        try:
            await self.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка отправки индикатора загрузки фото в чат {chat_id}: {e}")
            return False
    
    def get_dispatcher(self) -> Dispatcher:
        """
        Получение экземпляра диспетчера.
        
        Returns:
            Dispatcher: Диспетчер
        """
        return self.dp
    
    def get_bot(self) -> Bot:
        """
        Получение экземпляра бота.
        
        Returns:
            Bot: Бот
        """
        return self.bot
    
    async def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """
        Получение информации о файле.
        
        Args:
            file_id: ID файла
            
        Returns:
            dict: Информация о файле
        """
        try:
            file = await self.bot.get_file(file_id)
            file_info = {
                "file_id": file.file_id,
                "file_unique_id": file.file_unique_id,
                "file_size": file.file_size,
                "file_path": file.file_path,
            }
            
            self.logger.debug(f"Информация о файле {file_id}: {file_info}")
            return file_info
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения информации о файле {file_id}: {e}")
            return {"file_id": file_id, "error": str(e)}
    
    async def delete_message_safe(self, chat_id: int, message_id: int) -> bool:
        """
        Безопасное удаление сообщения.
        
        Args:
            chat_id: ID чата
            message_id: ID сообщения
            
        Returns:
            bool: True если сообщение удалено успешно
        """
        try:
            await self.bot.delete_message(chat_id=chat_id, message_id=message_id)
            self.logger.debug(f"✅ Сообщение {message_id} удалено из чата {chat_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка удаления сообщения {message_id} из чата {chat_id}: {e}")
            return False
    
    async def edit_message_safe(self, chat_id: int, message_id: int, text: str, **kwargs) -> bool:
        """
        Безопасное редактирование сообщения.
        
        Args:
            chat_id: ID чата
            message_id: ID сообщения
            text: Новый текст сообщения
            **kwargs: Дополнительные параметры
            
        Returns:
            bool: True если сообщение отредактировано успешно
        """
        try:
            await self.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                **kwargs
            )
            self.logger.debug(f"✅ Сообщение {message_id} отредактировано в чате {chat_id}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Ошибка редактирования сообщения {message_id} в чате {chat_id}: {e}")
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса менеджера бота.
        
        Returns:
            dict: Статус бота и диспетчера
        """
        try:
            bot_info = await self.get_bot_info()
            
            return {
                "bot_connected": bool(bot_info),
                "bot_username": bot_info.get("username", "unknown"),
                "bot_id": bot_info.get("id", "unknown"),
                "dispatcher_running": self.dp is not None,
                "commands_count": len(BOT_COMMANDS)
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка получения статуса: {e}")
            return {
                "bot_connected": False,
                "error": str(e)
            }
    
    async def cleanup(self):
        """
        Очистка ресурсов при остановке бота.
        """
        try:
            self.logger.info("🧹 Начинаем очистку ресурсов бота...")
            
            # Закрываем сессию бота
            if hasattr(self.bot, 'session') and self.bot.session:
                await self.bot.session.close()
                self.logger.debug("✅ Сессия бота закрыта")
            
            # Очищаем кэш информации о боте
            self.bot_info = None
            
            self.logger.info("✅ Ресурсы бота очищены")
        except Exception as e:
            self.logger.error(f"❌ Ошибка при очистке ресурсов: {e}")
    
    def __str__(self) -> str:
        """Строковое представление менеджера бота."""
        if self.bot_info:
            return f"BotManager(@{self.bot_info.get('username', 'unknown')})"
        return "BotManager(not_initialized)"
    
    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"BotManager(bot={self.bot}, dp={self.dp})"
        