# Birthday Bot Makefile
# Удобные команды для разработки и развертывания

# Переменные
ENV_NAME = birthday_bot
PYTHON = python
PIP = pip

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m

.PHONY: help setup install clean run dev test lint format check activate info

# Команда по умолчанию
help:
	@echo "$(GREEN)🎉 Birthday Bot - Доступные команды:$(NC)"
	@echo ""
	@echo "  $(YELLOW)setup$(NC)     - Полная настройка проекта (conda + зависимости)"
	@echo "  $(YELLOW)install$(NC)   - Установка зависимостей в существующее окружение"
	@echo "  $(YELLOW)run$(NC)       - Запуск бота"
	@echo "  $(YELLOW)dev$(NC)       - Запуск в режиме разработки (с watchdog)"
	@echo "  $(YELLOW)test$(NC)      - Запуск тестов"
	@echo "  $(YELLOW)lint$(NC)      - Проверка кода (flake8)"
	@echo "  $(YELLOW)format$(NC)    - Форматирование кода (black)"
	@echo "  $(YELLOW)check$(NC)     - Проверка конфигурации"
	@echo "  $(YELLOW)clean$(NC)     - Очистка временных файлов"
	@echo "  $(YELLOW)info$(NC)      - Информация о проекте"
	@echo ""
	@echo "$(GREEN)Перед запуском активируйте окружение:$(NC)"
	@echo "  conda activate $(ENV_NAME)"

# Полная настройка проекта
setup:
	@echo "$(GREEN)🏗️ Настройка проекта Birthday Bot...$(NC)"
	@chmod +x setup.sh
	@./setup.sh

# Установка зависимостей в активное окружение
install:
	@echo "$(GREEN)📦 Установка зависимостей...$(NC)"
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)✅ Зависимости установлены!$(NC)"

# Запуск бота
run:
	@echo "$(GREEN)🚀 Запуск Birthday Bot...$(NC)"
	@$(PYTHON) src/main.py

# Запуск в режиме разработки
dev:
	@echo "$(GREEN)🔧 Запуск в режиме разработки...$(NC)"
	@$(PYTHON) run_bot.py --debug

# Проверка конфигурации
check:
	@echo "$(GREEN)🔍 Проверка конфигурации...$(NC)"
	@$(PYTHON) src/main.py --validate-only

# Запуск тестов
test:
	@echo "$(GREEN)🧪 Запуск тестов...$(NC)"
	@$(PYTHON) -m pytest tests/ -v

# Проверка кода
lint:
	@echo "$(GREEN)🔍 Проверка кода с flake8...$(NC)"
	@$(PYTHON) -m flake8 src/ --max-line-length=88 --extend-ignore=E203,W503

# Форматирование кода
format:
	@echo "$(GREEN)🎨 Форматирование кода с black...$(NC)"
	@$(PYTHON) -m black src/ --line-length=88

# Проверка типов
mypy:
	@echo "$(GREEN)🔍 Проверка типов с mypy...$(NC)"
	@$(PYTHON) -m mypy src/ --ignore-missing-imports

# Очистка временных файлов
clean:
	@echo "$(GREEN)🧹 Очистка временных файлов...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf temp/audio/* temp/images/* logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✅ Временные файлы очищены!$(NC)"

# Информация о проекте
info:
	@echo "$(GREEN)📋 Информация о проекте Birthday Bot$(NC)"
	@echo ""
	@echo "Версия Python: $$(python --version 2>&1)"
	@echo "Активное окружение: $${CONDA_DEFAULT_ENV:-none}"
	@echo "Рабочая директория: $$(pwd)"
	@echo ""
	@echo "Структура проекта:"
	@tree -L 2 -I '__pycache__|*.pyc|*.egg-info' . 2>/dev/null || find . -type d -name "__pycache__" -prune -o -type d -print | head -20
	@echo ""
	@echo "Размер временных файлов:"
	@du -sh temp/ logs/ 2>/dev/null || echo "Временные директории пусты"

# Создание архива проекта
archive:
	@echo "$(GREEN)📦 Создание архива проекта...$(NC)"
	@tar -czf birthday_bot_$$(date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='temp/*' \
		--exclude='logs/*' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.env' \
		.
	@echo "$(GREEN)✅ Архив создан!$(NC)"

# Быстрая проверка всего проекта
verify: lint mypy check
	@echo "$(GREEN)✅ Все проверки пройдены!$(NC)"

# Обновление зависимостей
upgrade:
	@echo "$(GREEN)⬆️ Обновление зависимостей...$(NC)"
	@$(PIP) install --upgrade -r requirements.txt

# Показать активные процессы бота
ps:
	@echo "$(GREEN)🔍 Активные процессы Birthday Bot:$(NC)"
	@ps aux | grep -E "(python.*main\.py|python.*run_bot\.py)" | grep -v grep || echo "Процессы не найдены"

# Остановка всех процессов бота
stop:
	@echo "$(YELLOW)⏹️ Остановка всех процессов Birthday Bot...$(NC)"
	@pkill -f "python.*main\.py" 2>/dev/null || true
	@pkill -f "python.*run_bot\.py" 2>/dev/null || true
	@echo "$(GREEN)✅ Процессы остановлены$(NC)"

# Перезапуск бота
restart: stop
	@sleep 2
	@$(MAKE) dev

# Мониторинг логов
logs:
	@echo "$(GREEN)📊 Мониторинг логов...$(NC)"
	@tail -f logs/*.log 2>/dev/null || echo "Логи не найдены. Запустите бота для создания логов."

# Показать последние ошибки
errors:
	@echo "$(RED)🚨 Последние ошибки:$(NC)"
	@tail -20 logs/error.log 2>/dev/null || echo "Файл ошибок не найден"

# Статистика использования
stats:
	@echo "$(GREEN)📈 Статистика использования:$(NC)"
	@echo ""
	@echo "Количество логов:"
	@wc -l logs/*.log 2>/dev/null || echo "Логи не найдены"
	@echo ""
	@echo "Размер временных файлов:"
	@find temp/ -type f -exec ls -lh {} \; 2>/dev/null | wc -l | xargs echo "Временных файлов:"
	@echo ""
	@echo "Последняя активность:"
	@ls -lt logs/*.log 2>/dev/null | head -3 || echo "Логи не найдены"
	