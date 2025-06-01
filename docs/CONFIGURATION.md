# Настройка конфигурации Birthday Bot

Система конфигурации использует многоуровневый подход с приоритетами для максимальной гибкости и безопасности.

## 📁 Структура конфигурации

```
birthday_bot/
├── conf/
│   └── config.yaml          # Основная конфигурация (в репозитории)
├── .env.example             # Пример переменных окружения
├── .env                     # Переменные окружения (создается пользователем)
└── ../../
    └── secrets.yaml         # Токены и секреты (ВНЕ репозитория)
```

## 🔄 Приоритет настроек

Настройки применяются в следующем порядке (каждый следующий переопределяет предыдущий):

1. **Значения по умолчанию** (в коде)
2. **conf/config.yaml** (основная конфигурация)
3. **../../secrets.yaml** (токены и переопределения)
4. **Переменные окружения** (высший приоритет)

## ⚙️ Настройка основной конфигурации

### 1. Основной файл конфигурации

Файл `conf/config.yaml` содержит все параметры проекта **кроме токенов**:

```yaml
# Пример настройки модели
diffusion:
  model: "stabilityai/stable-diffusion-xl-base-1.0"
  device: "auto"
  width: 1024
  height: 1024
  num_inference_steps: 28

speech:
  model_name: "small"
  device: "cpu"
  language: "ru"
```

**Преимущества:**
- ✅ Находится в репозитории
- ✅ Версионируется в Git
- ✅ Безопасен (нет секретов)
- ✅ Документирован

## 🔐 Настройка токенов и секретов

### 2. Файл с секретами

Создайте файл `../../secrets.yaml` **за пределами репозитория**:

```yaml
# Минимальная конфигурация
bot:
  token: "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"

# Дополнительные переопределения (опционально)
diffusion:
  device: "cuda"  # Принудительное использование GPU

speech:
  model_name: "medium"  # Лучшее качество

logging:
  level: "DEBUG"  # Отладочный режим
```

**Безопасность:**
- ✅ За пределами репозитория
- ✅ Не попадает в Git
- ✅ Персональные настройки
- ✅ Может переопределять любые параметры

### 3. Переменные окружения

Создайте файл `.env` для локальных переопределений:

```bash
# Копируйте из .env.example
cp .env.example .env

# Редактируйте под свои нужды
TELEGRAM_BOT_TOKEN=your_token_here
LOG_LEVEL=DEBUG
DIFFUSION_DEVICE=cuda
```

## 🚀 Быстрая настройка

### Минимальная настройка

1. **Создайте файл с токеном:**
```bash
mkdir -p ../../
cat > ../../secrets.yaml << EOF
bot:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
EOF
```

2. **Запустите бота:**
```bash
python src/main.py
```

### Расширенная настройка

1. **Настройте основную конфигурацию:**
```bash
# Отредактируйте conf/config.yaml под свои нужды
nano conf/config.yaml
```

2. **Создайте персональные настройки:**
```bash
cat > ../../secrets.yaml << EOF
bot:
  token: "YOUR_TELEGRAM_BOT_TOKEN"

# Ваши переопределения
diffusion:
  device: "cuda"
  model: "runwayml/stable-diffusion-v1-5"

speech:
  model_name: "medium"
  device: "cuda"

logging:
  level: "DEBUG"
EOF
```

3. **Настройте переменные окружения (опционально):**
```bash
cp .env.example .env
nano .env
```

## 📝 Примеры конфигураций

### Для разработки

```yaml
# ../../secrets.yaml
bot:
  token: "YOUR_DEV_BOT_TOKEN"

logging:
  level: "DEBUG"

development:
  auto_reload: true
  verbose_console: true
  save_debug_images: true
```

### Для продакшена

```yaml
# ../../secrets.yaml
bot:
  token: "YOUR_PROD_BOT_TOKEN"

logging:
  level: "INFO"

production:
  optimize_memory: true
  cleanup_on_start: true
  performance_monitoring: true

diffusion:
  device: "cuda"
  preload_model: true
```

### Для слабых устройств

```yaml
# ../../secrets.yaml
bot:
  token: "YOUR_BOT_TOKEN"

diffusion:
  model: "runwayml/stable-diffusion-v1-5"  # Меньшая модель
  device: "cpu"
  width: 512
  height: 512
  num_inference_steps: 20

speech:
  model_name: "base"  # Быстрая модель
```

## 🔍 Проверка конфигурации

### Валидация настроек

```bash
# Проверка конфигурации без запуска
python src/main.py --validate-only

# Отладочная информация
python src/main.py --debug --validate-only
```

### Просмотр статуса

```python
from src.utils.config import config

# Проверка статуса
status = config.get_status()
print(status)

# Проверка ошибок
errors = config.validate()
for error in errors:
    print(error)
```

## 🛠️ Устранение проблем

### Токен не найден

```bash
❌ Не указан токен Telegram бота
```

**Решение:** Создайте файл `../../secrets.yaml` с токеном:
```yaml
bot:
  token: "YOUR_TELEGRAM_BOT_TOKEN"
```

### Модель не найдена

```bash
❌ Неверная модель Whisper: invalid_model
```

**Решение:** Используйте корректное имя модели в `conf/config.yaml`:
```yaml
speech:
  model_name: "small"  # tiny, base, small, medium, large
```

### CUDA недоступна

```bash
❌ CUDA недоступна, но указана в device
```

**Решение:** Измените устройство на CPU:
```yaml
diffusion:
  device: "cpu"  # или "auto"
```

## 📚 Справочник настроек

### Основные секции

- **`bot`** - настройки Telegram бота
- **`speech`** - настройки распознавания речи (Whisper)
- **`diffusion`** - настройки генерации изображений (Stable Diffusion)
- **`security`** - лимиты и безопасность
- **`paths`** - пути к файлам и директориям
- **`logging`** - настройки логирования

### Переменные окружения

- **`TELEGRAM_BOT_TOKEN`** - токен бота
- **`LOG_LEVEL`** - уровень логирования
- **`WHISPER_MODEL`** - модель Whisper
- **`DIFFUSION_MODEL`** - модель Stable Diffusion
- **`DIFFUSION_DEVICE`** - устройство для генерации
- **`ENVIRONMENT`** - режим работы (development/production)

## 🔒 Безопасность

### Что НЕ добавлять в Git

❌ **Никогда не добавляйте в репозиторий:**
- Файлы с токенами (`secrets.yaml`)
- Приватные `.env` файлы
- API ключи
- Пароли

✅ **Безопасно добавлять:**
- `conf/config.yaml` (без секретов)
- `.env.example` (примеры без реальных значений)
- Документацию

### Файл .gitignore

Убедитесь, что `.gitignore` содержит:
```gitignore
# Секреты и токены
secrets.yaml
*.secret.yaml
.env
.env.local

# Временные файлы
temp/
logs/
```
