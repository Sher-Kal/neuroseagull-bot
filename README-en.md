# 🎭 NeuroSeagull Bot

Telegram bot for actors of a theatre:
- 🎟 Quick lookup of ticket sales by show code
- 🤖 Text chat with GPT (RU persona, sarcastic/aphoristic)
- 🎨 Image generation (DALL·E)
- 🎬 Silent short video (text-to-video with Replicate)
- 🎙 Voice input and TTS reply

## 🚀 Requirements
- Python 3.10+
- Google Chrome + chromedriver (in PATH)
- Dependencies in `requirements.txt`

Install deps:
```bash
pip install -r requirements.txt
```

## ⚙️ Config
Create `.env` (see `.env.example`):
```
TELEGRAM_TOKEN=...
OPENAI_API_KEY=...
REPLICATE_API_TOKEN=...
THEATER_EMAIL=...
THEATER_PASSWORD=...
```

## ▶️ Run
```bash
python seagullbot.py
```

Or via Supervisor (recommended for server deployment):
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

## 🔒 Notes
- Access to theatre admin panel must be authorized by the theatre itself.
- Bot is intended for internal use by actors/employees.

## 📜 License
MIT
