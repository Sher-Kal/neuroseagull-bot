# 🎭 Нейрочайка

Телеграм-бот для актёров театра:

- 🎟 Быстрый просмотр продаж по коду спектакля  
- 🤖 Текстовый чат с GPT (русскоязычная персона, ироничный тон)  
- 🎨 Генерация картинок (DALL·E)  
- 🎬 Короткие беззвучные видео (text-to-video через Replicate)  
- 🎙 Обмен голосовыми: распознавание речи и синтез ответа  

## 🚀 Требования
- Python 3.10+  
- Google Chrome + chromedriver (в PATH)  
- Зависимости из `requirements.txt`

Установка зависимостей:
```bash
pip install -r requirements.txt
```

## ⚙️ Конфиг
Создайте `.env` (см. `.env.example`):
```
TELEGRAM_TOKEN=...
OPENAI_API_KEY=...
REPLICATE_API_TOKEN=...
THEATER_EMAIL=...
THEATER_PASSWORD=...
```

## ▶️ Запуск
```bash
python seagullbot.py
```

Или через Supervisor (рекомендуется для сервера):
```
[program:neuroseagull]
command=/usr/bin/env python3 /opt/neuroseagull-bot/seagullbot.py
directory=/opt/neuroseagull-bot
autostart=true
autorestart=true
stopsignal=TERM
environment=TELEGRAM_TOKEN="...",OPENAI_API_KEY="...",REPLICATE_API_TOKEN="...",THEATER_EMAIL="...",THEATER_PASSWORD="..."
stdout_logfile=/var/log/neuroseagull.out.log
stderr_logfile=/var/log/neuroseagull.err.log
```

## 🔒 Примечания
- Доступ к админ-панели театра должен быть официально разрешён самим театром.  
- Бот предназначен для внутреннего использования сотрудниками/актёрами.  

## 📜 Лицензия
MIT
