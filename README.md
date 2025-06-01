# 🎉 Birthday Bot с AI-генерацией изображений

Современный Telegram бот для создания поздравительных картинок с днем рождения. Использует передовые технологии распознавания речи OpenAI Whisper и AI-генерации изображений через Hugging Face Flux Dev для создания красивых персонализированных поздравлений.

## ✨ Возможности

- 🎤 **Распознавание речи** - конвертация голосовых сообщений в текст с помощью OpenAI Whisper
- 🤖 **AI-генерация изображений** - создание уникальных поздравительных картинок с помощью Hugging Face Flux Dev
- 📝 **Обработка текста** - работа с текстовыми сообщениями любой длины
- 🎨 **Умное наложение текста** - автоматическое добавление поздравительного текста на AI-изображения
- 🔄 **Автоматический перезапуск** - мониторинг изменений файлов с помощью watchdog
- 📊 **Подробное логирование** - модульная система логирования с ротацией файлов
- ⚡ **Асинхронная обработка** - высокая производительность с множественными запросами
- 🛡️ **Безопасность** - rate limiting, валидация файлов, безопасное хранение токенов
- 🔧 **Удобная разработка** - автоматическая настройка окружения и удобные команды

## 🚀 Быстрый старт

### 1. Создание бота в Telegram

1. Найдите [@BotFather](https://t.me/BotFather) в Telegram
2. Отправьте команду `/newbot`
3. Следуйте инструкциям для создания бота
4. Сохраните полученный токен

### 2. Получение Hugging Face API ключа

1. Зарегистрируйтесь на [Hugging Face](https://huggingface.co)
2. Перейдите в [настройки токенов](https://huggingface.co/settings/tokens)
3. Создайте новый токен с правами на чтение
4. Сохраните токен для настройки

### 3. Автоматическая установка

```bash
# Клонируйте репозиторий
git clone <repository-url>
cd birthday_bot

# Запустите автоматическую настройку
chmod +x setup.py
python setup.py

# Или используйте готовые скрипты
./start_dev.sh    # Для разработки
./start_prod.sh   # Для продакшена
```

### 4. Настройка токенов

#### Способ 1: Внешний файл (рекомендуется)
```bash
# Отредактируйте конфигурационный файл
nano ~/.ssh/bot.yaml
```

```yaml
bot:
  token: YOUR_TELEGRAM_BOT_TOKEN_HERE

huggingface:
  api_key: YOUR_HUGGINGFACE_API_KEY_HERE
  model: black-forest-labs/FLUX.1-dev
```

#### Способ 2: Переменные окружения
```bash
# Отредактируйте .env файл
nano .env
```

```bash
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### 5. Запуск

```bash
# Активируйте conda окружение
conda activate amikhalev_hb_2025_06

# Запуск в режиме разработки (рекомендуется)
./start_dev.sh
# или
python run_bot.py --debug

# Запуск в продакшене
./start_prod.sh
# или
python src/main.py
```

## 📁 Архитектура проекта

### Структура модулей
```
src/
├── main.py                 # Точка входа с валидацией и запуском
├── bot/
│   ├── bot_instance.py     # BotManager - управление ботом
│   └── handlers.py         # Обработчики команд и сообщений
├── speech/
│   ├── speech_to_text.py   # Whisper интеграция
│   └── audio_processor.py  # FFmpeg обработка аудио
├── image/
│   └── generator.py        # AI-генерация изображений (Hugging Face + Pillow)
└── utils/
    ├── config.py           # Централизованная конфигурация
    └── logger.py           # Модульное логирование
```

### Принципы архитектуры
- **Модульность** - независимые компоненты с четкими границами
- **Асинхронность** - полная поддержка async/await
- **AI-интеграция** - современные модели для генерации изображений
- **Fallback система** - автоматический переход к простым изображениям при сбоях API
- **Конфигурируемость** - гибкая настройка через файлы и переменные
- **Отказоустойчивость** - комплексная обработка ошибок

## ⚙️ Конфигурация

### Модели и сервисы

#### Hugging Face Flux Dev
- **Модель**: `black-forest-labs/FLUX.1-dev`
- **Качество**: Высочайшее качество AI-генерации
- **Скорость**: 20-60 секунд на изображение
- **Требования**: Hugging Face API ключ (бесплатный)

#### Модели Whisper

| Модель | Размер | Скорость | Качество | Использование |
|--------|--------|----------|----------|---------------|
| `tiny` | ~39 MB | ⚡⚡⚡ | ⭐⭐ | Быстрое тестирование |
| `base` | ~74 MB | ⚡⚡ | ⭐⭐⭐ | Разработка |
| `small` | ~244 MB | ⚡ | ⭐⭐⭐⭐ | **Рекомендуется** |
| `medium` | ~769 MB | 🐌 | ⭐⭐⭐⭐⭐ | Высокое качество |
| `large` | ~1550 MB | 🐌🐌 | ⭐⭐⭐⭐⭐ | Максимальное качество |

### Основные настройки

```bash
# .env файл
TELEGRAM_BOT_TOKEN=your_bot_token
HUGGINGFACE_API_KEY=your_hf_key
HUGGINGFACE_MODEL=black-forest-labs/FLUX.1-dev
WHISPER_MODEL=small              # Модель для распознавания
WHISPER_DEVICE=cpu               # cpu или cuda
LOG_LEVEL=INFO                   # Уровень логирования
MAX_VOICE_DURATION=60            # Макс. длительность голосовых (сек)
RATE_LIMIT_MESSAGES=10           # Лимит сообщений в минуту
```

## 🎯 Использование

### Команды бота

- `/start` - Начать работу с ботом и получить инструкции
- `/help` - Подробная справка по функциям

### Поддерживаемые форматы

**Текстовые сообщения**: Любой текст поздравления  
**Голосовые сообщения**: OGG, MP3, WAV, M4A, FLAC (до 60 сек)

### Примеры взаимодействия

```
👤 Пользователь: /start
🤖 Бот: 🎉 Добро пожаловать в Birthday Bot!
      Отправьте текст или запишите голосовое сообщение...

👤 Пользователь: "С днем рождения! Желаю счастья, здоровья и успехов!"
🤖 Бот: ⏳ Обрабатываю ваше сообщение...
      🎨 Создаю поздравительную картинку...
      [Отправляет уникальное AI-изображение с наложенным текстом]

👤 Пользователь: [Голосовое: "Поздравляю с праздником! Пусть сбудутся все мечты!"]
🤖 Бот: 🎤 Распознаю голосовое сообщение...
      ✅ Распознанный текст: "Поздравляю с праздником! Пусть сбудутся все мечты!"
      🎨 Создаю поздравительную картинку...
      [Отправляет персонализированное AI-изображение]
```

## 🎨 Технологии генерации изображений

### AI-генерация (основной режим)
- **Hugging Face Flux Dev** - современная модель генерации изображений
- **Умные промпты** - автоматическое создание описаний на основе текста поздравления
- **Контекстные ключевые слова** - извлечение тематики из русского текста
- **Наложение текста** - красивое добавление поздравления на AI-изображение

### Fallback система
- **Градиентные фоны** - красивые переходы цветов при недоступности AI
- **Декоративные элементы** - автоматические украшения и эффекты
- **Адаптивный текст** - автоматический перенос и центрирование
- **Надежность** - гарантированная генерация изображения в любых условиях

## 🔧 Разработка

### Команды разработки

```bash
# Проверка конфигурации
python src/main.py --validate-only

# Разработка с автоперезапуском
python run_bot.py --debug

# Запуск в продакшене
python src/main.py

# Просмотр логов
tail -f logs/*.log

# Очистка временных файлов
find temp/ -type f -name "*.png" -mtime +1 -delete
```

### Структура логов

```
logs/
├── birthday_bot_main.log      # Основное приложение
├── birthday_bot_bot.log       # Telegram бот и обработчики
├── birthday_bot_speech.log    # Распознавание речи
├── birthday_bot_image.log     # Генерация изображений
├── birthday_bot_handlers.log  # Обработчики сообщений
└── error.log                  # Консолидированные ошибки
```

### AI-генерация: детали работы

```python
# Пример создания промпта
def create_birthday_prompt(text):
    base_prompt = (
        "A beautiful birthday celebration scene with festive decorations, "
        "colorful balloons, confetti, birthday cake with candles, "
        "warm lighting, cheerful atmosphere, high quality, detailed"
    )
    
    # Извлечение ключевых слов из русского текста
    keywords = extract_keywords(text)  # цветы -> flowers, торт -> cake
    
    return f"{base_prompt}, {', '.join(keywords)}, digital art, photorealistic"
```

## 📈 Мониторинг и статистика

Бот автоматически отслеживает:

- 👤 **Действия пользователей** - команды, сообщения, время обработки
- 🎤 **Качество распознавания** - длительность аудио, время обработки, успешность
- 🤖 **AI-генерация** - время создания, успешность, fallback использование
- 🎨 **Обработка изображений** - размер текста, время наложения, качество
- ❌ **Ошибки с контекстом** - детальная информация для быстрого решения
- 📊 **Производительность** - метрики каждого этапа обработки

## 🛠️ Устранение проблем

### Частые проблемы и решения

#### 1. Hugging Face API ошибки
```bash
# Проверка API ключа
curl -H "Authorization: Bearer YOUR_HF_TOKEN" \
     https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev

# Проверка квоты
# Бесплатный план: 1000 запросов в месяц
# Про план: больше запросов и скорость
```

#### 2. Модель загружается (503 ошибка)
```bash
# Это нормально для первого запроса
# Модель автоматически загружается на серверах HF
# Обычно занимает 20-60 секунд
# Бот автоматически повторит запрос
```

#### 3. FFmpeg не найден
```bash
# Проверка установки
ffmpeg -version

# Ubuntu/Debian
sudo apt install ffmpeg

# Или переустановка через conda
conda install -c conda-forge ffmpeg
```

#### 4. Модель Whisper не загружается
```bash
# Очистка кэша
pip cache purge
conda activate amikhalev_hb_2025_06
pip install --no-cache-dir openai-whisper

# Проверка доступных моделей
python -c "import whisper; print(whisper.available_models())"
```

#### 5. Медленная генерация изображений
```bash
# Возможные причины:
# 1. Модель загружается (первый запрос)
# 2. Высокая нагрузка на серверы HF
# 3. Бесплатный план HF (медленнее Pro)

# Решения:
# 1. Подождать загрузки модели
# 2. Повторить запрос позже
# 3. Рассмотреть Pro план HF
```

### Диагностика системы

```bash
# Полная проверка системы
python src/main.py --validate-only

# Проверка Hugging Face подключения
python -c "
from src.utils.config import config
print('HF API key configured:', bool(config.huggingface.api_key))
print('HF model:', config.huggingface.model)
"

# Проверка conda окружения
conda info
conda list | grep -E "(torch|transformers|diffusers|huggingface)"

# Проверка дискового пространства
df -h temp/ logs/
```

## 🚀 Развертывание в продакшене

### Системные требования

- **ОС**: Linux (Ubuntu 20.04+, CentOS 8+) или macOS
- **RAM**: минимум 4GB (рекомендуется 8GB для стабильной работы)
- **Диск**: 10GB свободного места (для моделей и временных файлов)
- **Интернет**: стабильное соединение для Hugging Face API
- **Python**: 3.9+
- **Conda**: последняя версия

### Настройка продакшена

```bash
# 1. Клонирование и настройка
git clone <repository-url> /opt/birthday_bot
cd /opt/birthday_bot
python setup.py

# 2. Настройка конфигурации для продакшена
sudo mkdir -p /opt/birthday_bot/.ssh
sudo tee /opt/birthday_bot/.ssh/bot.yaml > /dev/null <<EOF
bot:
  token: YOUR_PRODUCTION_TELEGRAM_TOKEN

huggingface:
  api_key: YOUR_PRODUCTION_HF_TOKEN
  model: black-forest-labs/FLUX.1-dev
  timeout: 120

whisper:
  model: small
  device: cpu

logging:
  level: INFO
EOF
sudo chmod 600 /opt/birthday_bot/.ssh/bot.yaml

# 3. Настройка systemd сервиса
sudo tee /etc/systemd/system/birthday-bot.service > /dev/null <<EOF
[Unit]
Description=Birthday Bot Telegram Service with AI Image Generation
After=network.target

[Service]
Type=simple
User=birthday-bot
WorkingDirectory=/opt/birthday_bot
Environment=PATH=/opt/miniconda3/envs/amikhalev_hb_2025_06/bin
ExecStart=/opt/miniconda3/envs/amikhalev_hb_2025_06/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# 4. Запуск сервиса
sudo systemctl daemon-reload
sudo systemctl enable birthday-bot
sudo systemctl start birthday-bot
```

### Мониторинг продакшена

```bash
# Статус сервиса
sudo systemctl status birthday-bot

# Логи сервиса
sudo journalctl -u birthday-bot -f

# Логи приложения
tail -f /opt/birthday_bot/logs/*.log

# Статистика использования ресурсов
htop
df -h
du -sh /opt/birthday_bot/temp/
```

## 📝 Roadmap

### Запланированные функции

- [ ] **Локальная AI-генерация** - интеграция с локальными моделями (Stable Diffusion)
- [ ] **Множественные стили** - различные художественные стили генерации
- [ ] **Персонализация** - сохранение предпочтений пользователей
- [ ] **Групповые чаты** - поддержка работы в группах Telegram
- [ ] **Web-панель администратора** - управление ботом через браузер
- [ ] **Статистика и аналитика** - детальная статистика использования
- [ ] **Множественные языки** - поддержка интерфейса на разных языках
- [ ] **API интеграция** - подключение других AI сервисов

### Улучшения производительности

- [ ] **GPU ускорение** - поддержка CUDA для локальной генерации
- [ ] **Кэширование изображений** - умное кэширование популярных запросов
- [ ] **CDN интеграция** - быстрая доставка результатов
- [ ] **Распределенная обработка** - масштабирование на несколько серверов
- [ ] **Оптимизация моделей** - квантизация и сжатие

## 💡 Примеры использования

### Тематические поздравления

```
"С днем рождения! Желаю море цветов и океан счастья!"
→ AI создает изображение с цветами и морской тематикой

"Поздравляю с праздником! Пусть торт будет сладким, а подарки - приятными!"
→ AI генерирует сцену с тортом и подарками

"С юбилеем! Желаю крепкого здоровья и семейного счастья!"
→ AI создает теплое семейное изображение
```

### Технические детали генерации

1. **Анализ текста** - извлечение ключевых слов и эмоций
2. **Создание промпта** - формирование описания для AI на английском
3. **AI-генерация** - отправка запроса к Hugging Face Flux Dev
4. **Постобработка** - наложение оригинального текста поздравления
5. **Оптимизация** - сжатие и оптимизация итогового изображения

## 🤝 Вклад в проект

### Как внести вклад

1. **Fork** проекта на GitHub
2. Создайте ветку для новой функции:
   ```bash
   git checkout -b feature/ai-image-styles
   ```
3. Разработайте и протестируйте изменения:
   ```bash
   python run_bot.py --debug
   python -m pytest tests/
   ```
4. Зафиксируйте изменения:
   ```bash
   git commit -m 'Add multiple AI image styles'
   ```
5. Отправьте в репозиторий:
   ```bash
   git push origin feature/ai-image-styles
   ```
6. Создайте **Pull Request**

### Стандарты кода

- **Форматирование**: используйте `black` для форматирования
- **Стиль**: следуйте `flake8` рекомендациям
- **Тесты**: добавляйте тесты для новых функций
- **Документация**: обновляйте README и комментарии
- **Типизация**: используйте type hints где возможно

## 📄 Лицензия

Этот проект распространяется под лицензией MIT. Подробности в файле `LICENSE`.

## 🙏 Благодарности

- **[Hugging Face](https://huggingface.co)** - за предоставление передовых AI моделей
- **[Flux Dev](https://github.com/black-forest-labs/flux)** - за революционную модель генерации изображений
- **[OpenAI Whisper](https://github.com/openai/whisper)** - за непревзойденное распознавание речи
- **[aiogram](https://github.com/aiogram/aiogram)** - за мощный и удобный фреймворк для Telegram ботов
- **[Pillow](https://python-pillow.org/)** - за всеобъемлющую работу с изображениями
- **[PyTorch](https://pytorch.org/)** - за лидирующую платформу машинного обучения

## 📞 Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-repo/birthday_bot/issues)
- **Документация**: См. папку `docs/`
- **Примеры**: См. `docs/examples/`
- **Hugging Face**: [Документация API](https://huggingface.co/docs/api-inference/index)

---

**Создавайте незабываемые AI-поздравления с Birthday Bot! 🎉✨🤖**
