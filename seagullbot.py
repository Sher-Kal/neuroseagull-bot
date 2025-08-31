#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NeuroSeagull (Нейрочайка)
Телеграм-бот для актёров театра:
— быстрый просмотр продаж по коду показа и по датам;
— чат с GPT (RU-персона);
— генерация картинок и коротких беззвучных видео (гифок).
"""

import os
import sys
import re
import atexit
import signal
import random
import logging
import tempfile
import traceback
from io import BytesIO
from datetime import datetime
from collections import defaultdict, deque

import telebot
from telebot import types
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from bs4 import BeautifulSoup

import openai
from openai import OpenAI
import replicate
from replicate.exceptions import ModelError

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация (через переменные окружения)
# ──────────────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "telebot-token")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "openai-api-key")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "api_token")
THEATER_EMAIL = os.getenv("THEATER_EMAIL", "theater-email")
THEATER_PASSWORD = os.getenv("THEATER_PASSWORD", "password")

# Модельки/параметры (можно менять без правки кода)
T2V_MODEL = os.getenv("T2V_MODEL", "wan-video/wan-2.2-t2v-fast")
IMG_MODEL = os.getenv("IMG_MODEL", "dall-e-3")
VOICE_TTS_MODEL = os.getenv("VOICE_TTS_MODEL", "gpt-4o-mini-tts")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4")

# ──────────────────────────────────────────────────────────────────────────────
# Логирование
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("neuroseagull")

# ──────────────────────────────────────────────────────────────────────────────
# Инициализация клиентов
# ──────────────────────────────────────────────────────────────────────────────

bot = telebot.TeleBot(TELEGRAM_TOKEN)
openai.api_key = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)
rep = replicate.Client(api_token=REPLICATE_API_TOKEN)

# ──────────────────────────────────────────────────────────────────────────────
# Глобальные вспомогательные структуры
# ──────────────────────────────────────────────────────────────────────────────

# Небольшая «память» диалога на пользователя (не база, просто последние ответы)
ASSIST_BY_USER: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=1))

seagull_dates: list[str] = []
seagull_codes: list[str] = []

THIS_MONTH = datetime.now().month
THIS_YEAR = datetime.now().year

# Отдельный профиль для headless-Chrome
USER_DATA_DIR = tempfile.mkdtemp(prefix="chrome-profile-")

# ──────────────────────────────────────────────────────────────────────────────
# Селениум: логин и драйвер (создаём один на весь процесс)
# ──────────────────────────────────────────────────────────────────────────────

def create_driver_logged_in() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("window-size=1920,1080")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    # профиль — чтобы меньше светиться как автотест
    opts.add_argument(f"--user-data-dir={USER_DATA_DIR}")
    # случайный порт для remote-debug
    opts.add_argument(f'--remote-debugging-port={random.randint(9200, 9400)}')

    drv = webdriver.Chrome(options=opts)
    drv.get("https://tickets.afisha.ru/admin/login")

    # Логин
    drv.find_element(By.ID, "email").send_keys(THEATER_EMAIL)
    drv.find_element(By.ID, "password").send_keys(THEATER_PASSWORD)
    drv.find_element(By.XPATH, "//button[contains(., 'Войти')]").click()

    # Подождём видимость главного блока кабинета
    WebDriverWait(drv, 20).until(
        ec.any_of(
            ec.url_contains("/admin"),
            ec.presence_of_element_located((By.TAG_NAME, "body"))
        )
    )
    return drv

driver = create_driver_logged_in()

def _shutdown_driver() -> None:
    try:
        driver.quit()
    except Exception:
        pass

atexit.register(_shutdown_driver)

def _graceful_exit(signum: int, frame) -> None:
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT, _graceful_exit)

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

MD_ESCAPE = str.maketrans({
    "_": r"\_",
    "*": r"\*",
    "[": r"\[",
    "]": r"\]",
    "(": r"\(",
    ")": r"\)",
    "~": r"\~",
    "`": r"\`",
    ">": r"\>",
    "#": r"\#",
    "+": r"\+",
    "-": r"\-",
    "=": r"\=",
    "|": r"\|",
    "{": r"\{",
    "}": r"\}",
    ".": r"\.",
    "!": r"\!",
    ",": r"\,",
    ":": r"\:",
})

def md_escape(text: str) -> str:
    return text.translate(MD_ESCAPE)

def get_user_assist(uid: int) -> str:
    dq = ASSIST_BY_USER[uid]
    return dq[-1] if dq else ""

def set_user_assist(uid: int, msg: str) -> None:
    ASSIST_BY_USER[uid].append(msg)

# ──────────────────────────────────────────────────────────────────────────────
# Голос ↔ текст
# ──────────────────────────────────────────────────────────────────────────────

def voice_to_text(path_ogg: str) -> str:
    with open(path_ogg, "rb") as f:
        tr = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f)
    return tr.text

def text_to_voice(text: str) -> BytesIO:
    # OpenAI TTS → opus/ogg
    rsp = client.audio.speech.create(
        input=text,
        model=VOICE_TTS_MODEL,
        voice="nova",
        response_format="opus",
    )
    bio = BytesIO(rsp.content)
    bio.seek(0)
    return bio

# ──────────────────────────────────────────────────────────────────────────────
# GPT-чат (RU-персона)
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_RU = (
    "Ты ассистентка, говоришь только по-русски. "
    "Имя — Панда или Елена или Нейрочайка; 19 лет, философски-саркастичный тон, афористично."
)

def gpt_reply(uid: int, text: str) -> str:
    try:
        assist_prev = get_user_assist(uid)
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_RU},
                {"role": "assistant", "content": assist_prev},
                {"role": "user", "content": text},
            ],
            max_tokens=1200,
            temperature=0.9,
        )
        msg = resp.choices[0].message.content
        set_user_assist(uid, msg)
        return msg
    except Exception as e:
        log.exception("OpenAI chat error")
        return "Сегодня язык не поворачивается… Спроси меня попозже."

# ──────────────────────────────────────────────────────────────────────────────
# Картинки и беззвучные видео
# ──────────────────────────────────────────────────────────────────────────────

def generate_image_from_prompt(prompt: str) -> str:
    try:
        rsp = client.images.generate(
            model=IMG_MODEL,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        return rsp.data[0].url
    except openai.BadRequestError:
        return ""
    except Exception:
        log.exception("Image generation error")
        return ""

def t2v_generate(
        prompt: str,
        aspect_ratio: str = "16:9",
        resolution: str = "480p",
        num_frames: int = 81,
        fps: int = 16,
        go_fast: bool = True,
):
    """
    Возвращает:
      • str (URL), если модель дала ссылку;
      • BytesIO — если пришёл бинарный поток.
    """
    try:
        out = rep.run(
            T2V_MODEL,
            input={
                "prompt": prompt,
                "go_fast": go_fast,
                "num_frames": num_frames,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "frames_per_second": fps,
            },
        )
        if hasattr(out, "url") and out.url:
            return out.url
        if hasattr(out, "read"):
            return BytesIO(out.read())
        if isinstance(out, (list, tuple)) and out:
            first = out[0]
            if hasattr(first, "url") and first.url:
                return first.url
            if hasattr(first, "read"):
                return BytesIO(first.read())
            if isinstance(first, str) and first.strip():
                return first.strip()
        if isinstance(out, str) and out.strip():
            return out.strip()
        raise RuntimeError("Replicate: пустой вывод модели")
    except ModelError as e:
        logs = ""
        try:
            logs = getattr(getattr(e, "prediction", None), "logs", "") or ""
        except Exception:
            pass
        raise RuntimeError(f"Replicate: ошибка модели. {logs or str(e)}")
    except Exception as e:
        raise RuntimeError(f"Replicate: непредвиденная ошибка: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Парсинг админки: продажа билетов
# ──────────────────────────────────────────────────────────────────────────────

def wait_ajax_complete(drv, timeout=20):
    WebDriverWait(drv, timeout).until(lambda d: d.execute_script(
        "return (document.readyState==='complete') && (window.jQuery ? jQuery.active==0 : true)"
    ))

def fetch_show_by_code(code: str) -> str:
    url = f"https://tickets.afisha.ru/admin/events/info/{code}"
    driver.get(url)
    wait_ajax_complete(driver, timeout=25)
    WebDriverWait(driver, 25).until(
        ec.visibility_of_element_located((By.CSS_SELECTOR, "div.pull-right.text-primary"))
    )
    soup = BeautifulSoup(driver.page_source, "html.parser")

    name = soup.find("a", href=re.compile(r"/admin/shows\?name="))
    element = soup.find("div", class_="pull-right text-primary")
    if not element or not name:
        return "Не получилось просканировать сайт. Попробуй ещё раз."

    date = element.get_text(strip=True).split(')')[1]
    places = soup.find_all("p", {"style": "margin-bottom: 4px;"})

    def _between(txt: str, left: str, right: str) -> str:
        try:
            return txt.split(left)[1].split(right)[0]
        except Exception:
            return "—"

    plains = str(places)
    sold_cnt = _between(plains, "Продано <b>", "</")
    sold_sum = _between(plains, "Продано <b>", "</b> р.").split("<b>")[-1].split(".")[0]
    sold_fact_cnt = _between(plains, "Продано фактически <b>", "</")
    sold_fact_sum = _between(plains, "Продано фактически <b>", "</b> р.").split("<b>")[-1]
    booked_cnt = _between(plains, "Забронировано <b>", "</")
    booked_sum = _between(plains, "Забронировано <b>", "</b> р.").split("<b>")[-1]

    return (
        f'Спектакль "{name.text}"\n'
        f"{date}\n"
        f"Продано билетов: {sold_cnt} на {sold_sum} рублей\n"
        f"Продано фактически: {sold_fact_cnt} на {sold_fact_sum} рублей\n"
        f"Забронировано: {booked_cnt} на {booked_sum} рублей"
    )

def fetch_month_menu(month_yyyy_mm: str, only_seagull: bool = False) -> tuple[list[str], list[str]]:
    """
    Возвращает (список строк-описаний, список URL-кодов).
    Если only_seagull=True — фильтруем по "ЧАЙКА".
    """
    url = f"https://tickets.afisha.ru/admin/events/menu_date?date={month_yyyy_mm}"
    driver.get(url)
    wait_ajax_complete(driver, timeout=25)
    WebDriverWait(driver, 25).until(
        ec.visibility_of_element_located((By.CSS_SELECTOR, "div.pull-right.text-primary"))
    )

    soup = BeautifulSoup(driver.page_source, "html.parser")
    blocks = soup.find_all(class_="nav navbar-nav extend-menu")
    if not blocks:
        return [], []

    soup2 = BeautifulSoup("".join(str(p) for p in blocks), "html.parser")
    links = soup2.find_all("a")

    items, codes = [], []
    for link in links:
        href = link.get("href", "")
        if not href.startswith("https://tickets.afisha.ru/admin/events/info/"):
            continue
        text = " ".join(link.stripped_strings)
        text = re.sub(r"\s+", " ", text)
        if only_seagull and "ЧАЙКА" not in text:
            continue
        items.append(text)
        codes.append(href.split("/info/")[1])

    return items, codes

# ──────────────────────────────────────────────────────────────────────────────
# Хэндлеры Telegram
# ──────────────────────────────────────────────────────────────────────────────

@bot.message_handler(commands=["start"])
def on_start(message: telebot.types.Message):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.row(types.KeyboardButton("🔢 Билеты по коду спектакля"))
    kb.row(types.KeyboardButton("📆 Узнать даты показа и коды"))
    kb.row(types.KeyboardButton("🔍 Проверить билеты \"Чайки\""))
    kb.row(types.KeyboardButton("🗄️ Прочее"))
    bot.send_message(message.chat.id, f"Привет, {message.from_user.first_name}!\nОриентируйся на меню ниже:", reply_markup=kb)

@bot.message_handler(func=lambda m: m.text and m.text.startswith("Сними: "))
def on_t2v(message: telebot.types.Message):
    prompt = message.text[len("Сними: "):].strip() or "A sports car driving on a beach at sunset"
    bot.send_chat_action(message.chat.id, "upload_video")
    try:
        result = t2v_generate(prompt, aspect_ratio="16:9", resolution="480p", num_frames=81, fps=16)
        if isinstance(result, str):
            bot.send_video(message.chat.id, result, caption=f"“{prompt}”")
        else:
            result.name = "video.mp4"
            bot.send_video(message.chat.id, result, caption=f"“{prompt}”")
    except Exception as e:
        bot.send_message(message.chat.id, f"Не получилось: {e}")

@bot.message_handler(func=lambda m: m.text and m.text.startswith("Нарисуй:"))
def on_image(message: telebot.types.Message):
    prompt = message.text[len("Нарисуй:"):].strip()
    url = generate_image_from_prompt(prompt)
    if not url:
        bot.send_message(message.chat.id, "Картинку сгенерировать не вышло.")
        return
    bot.send_message(message.chat.id, "Вот картинка по твоему запросу:")
    bot.send_photo(message.chat.id, url)

@bot.message_handler(content_types=["voice"])
def on_voice(message: telebot.types.Message):
    file_info = bot.get_file(message.voice.file_id)
    data = bot.download_file(file_info.file_path)
    tmp_path = "voice_message.ogg"
    with open(tmp_path, "wb") as f:
        f.write(data)
    try:
        text = voice_to_text(tmp_path)
        reply = gpt_reply(message.from_user.id, text)
        voice = text_to_voice(reply)
        bot.send_voice(message.chat.id, voice)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@bot.message_handler(content_types=["text"])
def on_text(message: telebot.types.Message):
    if message.chat.type != "private":
        return

    t = message.text.strip()

    if t == "🔢 Билеты по коду спектакля":
        bot.send_message(message.chat.id, "Введи код:")
        bot.register_next_step_handler(message, _ask_code)
        return

    if t == "📆 Узнать даты показа и коды":
        kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
        kb.row(types.KeyboardButton("🕰️ Спектакли в текущем месяце"))
        kb.row(types.KeyboardButton("🗂️ Спектакли в другом месяце"))
        kb.row(types.KeyboardButton("⬅️ Назад в главное меню"))
        bot.send_message(message.chat.id, "Выберите пункт:", reply_markup=kb)
        return

    if t == "🕰️ Спектакли в текущем месяце":
        month = f"{THIS_MONTH:02d}.{THIS_YEAR}"
        _send_month_list(message.chat.id, month)
        return

    if t == "🗂️ Спектакли в другом месяце":
        bot.send_message(message.chat.id, "Введи месяц и год в формате \"мм.гггг\". Например, 09.2025")
        bot.register_next_step_handler(message, _ask_month)
        return

    if t == "🔍 Проверить билеты \"Чайки\"":
        if not seagull_dates:
            bot.send_message(
                message.chat.id,
                "У бота пока нет дат\\! Нажми на кнопку \n`📥 Найти и добавить даты \"Чайки\"`\nв дополнительном меню\\.",
                parse_mode="MarkdownV2",
            )
        else:
            keyboard = types.InlineKeyboardMarkup()
            for date in seagull_dates:
                keyboard.add(types.InlineKeyboardButton(text=date, callback_data=date))
            bot.send_message(message.chat.id, "Выбери дату:", reply_markup=keyboard)
        return

    if t == "🗄️ Прочее":
        kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
        kb.row(types.KeyboardButton("📥 Найти и добавить даты \"Чайки\""))
        kb.row(types.KeyboardButton("🔄 Перезагрузить бота"))
        kb.row(types.KeyboardButton("📓 Информация"))
        kb.row(types.KeyboardButton("⬅️ Назад в главное меню"))
        bot.send_message(message.chat.id, "Выберите пункт:", reply_markup=kb)
        return

    if t == "⬅️ Назад в главное меню":
        return on_start(message)

    if t == "📥 Найти и добавить даты \"Чайки\"":
        _quick_add_seagull(message.chat.id)
        return

    if t == "🔄 Перезагрузить бота":
        bot.send_message(message.chat.id, "Бот перезагружается, подожди пару секунд и пользуйся снова")
        os.kill(os.getpid(), signal.SIGTERM)
        return

    if t == "📓 Информация":
        info = (
            "Краткая инструкция.\n\n"
            "🔍 Проверить билеты \"Чайки\" — быстрая проверка продаж.\n"
            "Если бот не знает даты показа, нажми: 📥 Найти и добавить даты \"Чайки\".\n\n"
            "Можно смотреть продажи любого спектакля по коду.\n"
            "Коды/даты ищутся через: 📆 Узнать даты показа и коды.\n"
            "Код копируется нажатием. Потом — в главное меню → 🔢 Билеты по коду спектакля.\n\n"
            "Обработка ошибок минимальная, вводите аккуратно.\n"
            "Онлайн-табличка \"График\" — кнопка слева от ввода текста (редактировать может любой).\n"
            "P.S. Прямая ссылка: https://disk.yandex.ru/i/RunXym0TQutCqA"
        )
        bot.send_message(message.chat.id, info)
        return

    # Иначе — простой диалог с GPT
    reply = gpt_reply(message.from_user.id, t)
    bot.send_message(message.chat.id, reply)

@bot.callback_query_handler(func=lambda call: True)
def on_choice(call: telebot.types.CallbackQuery):
    # Клик по дате из списка "Чайки"
    for date, code in zip(seagull_dates, seagull_codes):
        if call.data == date:
            bot.answer_callback_query(call.id)
            txt = fetch_show_by_code(code)
            bot.send_message(call.message.chat.id, txt)
            return

def _ask_code(message: telebot.types.Message):
    code = (message.text or "").strip()
    if not code.isdigit():
        bot.send_message(message.chat.id, "Нужен числовой код показа.")
        return
    bot.send_message(message.chat.id, "Смотрю билеты…")
    txt = fetch_show_by_code(code)
    bot.send_message(message.chat.id, txt)

def _ask_month(message: telebot.types.Message):
    month = (message.text or "").strip()
    if not re.fullmatch(r"\d{2}\.\d{4}", month):
        bot.send_message(message.chat.id, "Формат: мм.гггг (например, 09.2025)")
        return
    _send_month_list(message.chat.id, month)

def _send_month_list(chat_id: int, month: str):
    bot.send_message(chat_id, "Ищу все спектакли выбранного месяца…")
    items, codes = fetch_month_menu(month, only_seagull=False)
    if not items:
        bot.send_message(chat_id, "Спектаклей не найдено.")
        return

    # Порциями, экранируя для MarkdownV2
    chunk = 20
    for i in range(0, len(items), chunk):
        block = ""
        for text, code in zip(items[i:i+chunk], codes[i:i+chunk]):
            # Пример исходной строки: " 12.09.2025 «ЧАЙКА». ... "
            date = text[4:15]
            title = text[16:]
            line = f"{date}\n\"{title}\"\nКод: `{code}`\n\n"
            block += md_escape(line)
        bot.send_message(chat_id, block, parse_mode="MarkdownV2")

def _quick_add_seagull(chat_id: int):
    global seagull_dates, seagull_codes
    bot.send_message(chat_id, "Ищу все даты «Чайки» и добавляю в быстрый доступ…")

    m = THIS_MONTH
    y = THIS_YEAR

    def mm_yyyy(mm: int, yy: int) -> str:
        return f"{mm:02d}.{yy}"

    months = [mm_yyyy(m, y)]
    if m == 12:
        months.append(mm_yyyy(1, y + 1))
    else:
        months.append(mm_yyyy(m + 1, y))

    found_dates, found_codes = [], []
    for mon in months:
        items, codes = fetch_month_menu(mon, only_seagull=True)
        for t, c in zip(items, codes):
            found_dates.append(t[4:15])
            found_codes.append(c)

    if not found_dates:
        bot.send_message(chat_id, "Дат на ближайшие пару месяцев не обнаружено…")
        return

    seagull_dates += found_dates
    seagull_codes += found_codes
    bot.send_message(chat_id, "Даты добавлены. Можно смотреть продажи!")

# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Не пропускать апдейты (важно для supervisor-рестартов)
    telebot.TeleBot._TeleBot__skip_updates = lambda self: None  # type: ignore
    log.info("NeuroSeagull started.")
    bot.polling(none_stop=True, timeout=60, long_polling_timeout=50)
