#!/bin/bash
# Скрипт автоматической настройки окружения Birthday Bot для Linux

set -e  # Выход при любой ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Функции для цветного вывода
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${PURPLE}ℹ️ $1${NC}"
}

# Проверка наличия команды
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Проверка conda
check_conda() {
    if command_exists conda; then
        print_success "Conda найдена: $(conda --version)"
        return 0
    else
        print_error "Conda не найдена!"
        echo "Установите Anaconda или Miniconda:"
        echo "  - Anaconda: https://www.anaconda.com/download"
        echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        echo ""
        echo "Для Ubuntu/Debian можно установить через:"
        echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        echo "  bash Miniconda3-latest-Linux-x86_64.sh"
        return 1
    fi
}

# Проверка системных зависимостей
check_system_deps() {
    print_header "🔍 ПРОВЕРКА СИСТЕМНЫХ ЗАВИСИМОСТЕЙ"
    
    # Проверка FFmpeg
    if command_exists ffmpeg; then
        print_success "FFmpeg найден: $(ffmpeg -version 2>&1 | head -n1)"
    else
        print_warning "FFmpeg не найден в системе"
        echo "Для установки FFmpeg:"
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg"
        echo "  CentOS/RHEL: sudo yum install ffmpeg"
        echo "  Arch Linux: sudo pacman -S ffmpeg"
        echo ""
        print_info "FFmpeg будет установлен через conda"
    fi
    
    # Проверка Git
    if command_exists git; then
        print_success "Git найден: $(git --version)"
    else
        print_warning "Git не найден - установите для работы с репозиториями"
        echo "  Ubuntu/Debian: sudo apt install git"
        echo "  CentOS/RHEL: sudo yum install git"
    fi
    
    # Проверка Python
    if command_exists python3; then
        print_success "Python3 найден: $(python3 --version)"
    else
        print_error "Python3 не найден!"
        return 1
    fi
    
    echo ""
}

# Создание conda окружения
create_conda_env() {
    print_header "🏗️ СОЗДАНИЕ CONDA ОКРУЖЕНИЯ"
    
    local env_name="birthday_bot"
    
    # Проверяем существование окружения
    if conda env list | grep -q "^${env_name}"; then
        print_warning "Окружение '${env_name}' уже существует!"
        read -p "Пересоздать окружение? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Удаление существующего окружения '${env_name}'..."
            conda env remove -n "$env_name" -y
        else
            print_info "Используем существующее окружение '${env_name}'"
            return 0
        fi
    fi
    
    # Проверяем наличие файла environment.yml
    if [[ ! -f "environment.yml" ]]; then
        print_error "Файл environment.yml не найден!"
        return 1
    fi
    
    # Создаем окружение
    print_info "Создание окружения '${env_name}' из environment.yml..."
    conda env create -f environment.yml
    print_success "Окружение '${env_name}' успешно создано!"
    echo ""
}

# Создание структуры проекта
create_project_structure() {
    print_header "📁 СОЗДАНИЕ СТРУКТУРЫ ПРОЕКТА"
    
    local directories=(
        "src"
        "src/bot"
        "src/speech"
        "src/image"
        "src/utils"
        "src/models"
        "docs"
        "docs/examples"
        "docs/examples/voice_examples"
        "docs/examples/text_examples"
        "docs/images"
        "docs/images/screenshots"
        "temp"
        "temp/audio"
        "temp/images"
        "logs"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_success "Создана директория: $dir"
        else
            print_info "Директория уже существует: $dir"
        fi
    done
    
    # Создаем .gitkeep файлы
    local gitkeep_dirs=("temp/audio" "temp/images" "logs")
    for dir in "${gitkeep_dirs[@]}"; do
        local gitkeep_file="$dir/.gitkeep"
        if [[ ! -f "$gitkeep_file" ]]; then
            touch "$gitkeep_file"
            print_success "Создан .gitkeep в: $dir"
        fi
    done
    
    echo ""
}

# Создание __init__.py файлов
create_init_files() {
    print_header "🐍 СОЗДАНИЕ __init__.py ФАЙЛОВ"
    
    local init_dirs=(
        "src"
        "src/bot"
        "src/speech"
        "src/image"
        "src/utils"
        "src/models"
    )
    
    for dir in "${init_dirs[@]}"; do
        local init_file="$dir/__init__.py"
        if [[ ! -f "$init_file" ]]; then
            touch "$init_file"
            print_success "Создан __init__.py в: $dir"
        else
            print_info "__init__.py уже существует в: $dir"
        fi
    done
    
    echo ""
}

# Настройка файла окружения
setup_env_file() {
    print_header "⚙️ НАСТРОЙКА ФАЙЛА ОКРУЖЕНИЯ"
    
    if [[ ! -f ".env" ]] && [[ -f ".env.example" ]]; then
        cp ".env.example" ".env"
        print_success "Файл .env создан из .env.example"
        print_warning "ВАЖНО: Отредактируйте файл .env и укажите ваш TELEGRAM_BOT_TOKEN!"
    elif [[ -f ".env" ]]; then
        print_info "Файл .env уже существует"
    else
        print_warning "Файл .env.example не найден, создание .env пропущено"
    fi
    
    echo ""
}

# Создание конфигурации бота
create_bot_config() {
    print_header "🤖 НАСТРОЙКА КОНФИГУРАЦИИ БОТА"
    
    local ssh_dir="$HOME/.ssh"
    local bot_config_file="$ssh_dir/bot.yaml"
    
    if [[ ! -f "$bot_config_file" ]]; then
        print_info "Создание примера конфигурации: $bot_config_file"
        
        # Создаем директорию если не существует
        mkdir -p "$ssh_dir"
        
        # Создаем файл конфигурации
        cat > "$bot_config_file" << 'EOF'
# Birthday Bot Configuration
bot:
  token: YOUR_BOT_TOKEN_HERE
  
# Замените YOUR_BOT_TOKEN_HERE на реальный токен вашего бота
# Получить токен можно у @BotFather в Telegram
EOF
        
        # Устанавливаем права доступа только для владельца
        chmod 600 "$bot_config_file"
        
        print_success "Файл конфигурации создан: $bot_config_file"
        print_warning "ВАЖНО: Отредактируйте файл и укажите ваш токен бота!"
    else
        print_info "Файл конфигурации уже существует: $bot_config_file"
    fi
    
    echo ""
}

# Инструкции по активации
print_activation_instructions() {
    print_header "🎯 ИНСТРУКЦИИ ПО АКТИВАЦИИ"
    echo ""
    echo "Для активации созданного окружения выполните:"
    echo -e "${GREEN}  conda activate birthday_bot${NC}"
    echo ""
    echo "Для запуска бота:"
    echo -e "${GREEN}  python run_bot.py${NC}"
    echo ""
    echo "Для запуска в режиме отладки:"
    echo -e "${GREEN}  python run_bot.py --debug${NC}"
    echo ""
    echo "Для проверки конфигурации:"
    echo -e "${GREEN}  python src/main.py --validate-only${NC}"
    echo ""
    echo -e "${YELLOW}ВАЖНО: Не забудьте:${NC}"
    echo "  1. Указать токен бота в .env или ~/.ssh/bot.yaml"
    echo "  2. Создать бота через @BotFather в Telegram"
    echo ""
}

# Основная функция
main() {
    print_header "🎉 УСТАНОВКА BIRTHDAY BOT"
    echo "Этот скрипт настроит полное окружение для Birthday Bot"
    echo ""
    
    # Проверка conda
    if ! check_conda; then
        exit 1
    fi
    
    # Проверка системных зависимостей
    check_system_deps
    
    # Создание conda окружения
    if ! create_conda_env; then
        print_error "Не удалось создать conda окружение!"
        exit 1
    fi
    
    # Создание структуры проекта
    create_project_structure
    
    # Создание __init__.py файлов
    create_init_files
    
    # Настройка файлов окружения
    setup_env_file
    
    # Создание конфигурации бота
    create_bot_config
    
    echo ""
    print_success "🎉 УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!"
    print_activation_instructions
}

# Обработка прерывания
trap 'echo -e "\n⏹️ Установка прервана пользователем"; exit 1' INT

# Запуск основной функции
main "$@"
