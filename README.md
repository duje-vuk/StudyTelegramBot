# StudyBot

A self-hosted spaced repetition system that turns dead time into study time. Upload source material, AI generates quiz questions, you vet them, and a Telegram bot delivers them to your phone on a scientifically-timed schedule.

**The loop:** Source material → AI-generated questions → human vetting → spaced repetition → Telegram push notifications → self-grading → interval adjustment → repeat.

## How It Works

1. **Upload** lecture notes, PDFs, or paste text through the web UI
2. **AI generates** quiz questions using Groq (`qwen/qwen3.6-27b` by default) — free tier
3. **Review and approve** questions before they enter rotation
4. **Study via Telegram** — the bot sends due questions on schedule, you grade yourself, and the SM-2 algorithm adjusts intervals

Everything runs as a single process. The Telegram bot is embedded in the FastAPI app, sharing the same database. Deploy it on a server and your phone gets study notifications 24/7 — no laptop required.

## Tech Stack

- **Backend:** Python, FastAPI, SQLite
- **Frontend:** Vanilla HTML/CSS/JS in `static/` (no build tools)
- **AI:** Groq API (free tier), `qwen/qwen3.6-27b` by default
- **Notifications:** python-telegram-bot
- **PDF extraction:** PyMuPDF
- **Scheduling:** SM-2 spaced repetition algorithm

## Setup

### 1. Clone and install

```bash
git clone https://github.com/duje-vuk/StudyTelegramBot.git
cd StudyTelegramBot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

- **`GROQ_API_KEY`** — Get one free at [console.groq.com](https://console.groq.com)
- **`TELEGRAM_BOT_TOKEN`** — Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
- **`TELEGRAM_CHAT_ID`** — Send a message to your bot, then visit `https://api.telegram.org/bot<your-token>/getUpdates` to find your chat ID. The bot ignores every other chat.
- **`GROQ_MODEL`** (optional) — Groq model to use, default `qwen/qwen3.6-27b`

### 3. Run

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

The Telegram bot starts automatically if the token and chat ID are configured. Without them, the web UI still works — you just won't get phone notifications.

## Telegram Commands

| Command  | Description                              |
|----------|------------------------------------------|
| `/study` | Start a study session (back-to-back)     |
| `/due`   | Check how many questions are due         |
| `/stats` | View study statistics                    |
| `/pause` | Stop the current study session           |
| `/mute`  | Silence notifications                    |
| `/unmute`| Resume notifications                     |
| `/reset` | Reset all SM-2 progress                  |

## Deployment

A systemd service file (`study-bot.service`) is included for running on a Linux server. Edit the `User` and `WorkingDirectory` fields, then:

```bash
sudo cp study-bot.service /etc/systemd/system/
sudo systemctl enable study-bot
sudo systemctl start study-bot
```

Access the web UI via SSH tunnel:

```bash
ssh -L 3000:localhost:3000 <user>@<server-ip>
```

Then open [http://localhost:3000](http://localhost:3000) on your local machine.

## Architecture

```
Browser (static/)            Telegram app
     │ /generate, /api/*          │ commands, buttons
     ▼                            ▼
app.py (FastAPI routes)     telegramBot.py
     │                            │
     ├── questionGenerator.py ──► Groq API, PDF reading
     ├── spacedRepetition.py  (SM-2 scheduling)
     └── database.py ─────────► SQLite (spaced_rep.db)
           config.py holds settings and keys
```

### Where the AI is used

| Step | Prompt | Output |
|------|--------|--------|
| Generate questions from each text chunk | `prompts/questionGenerator.txt` | JSON list of questions |
| Name the topic of a source | `prompts/topicName.txt` | Short topic name |

Every AI reply is validated (`cleanQuestions`): malformed items are dropped and long text is trimmed. Generated questions start as *pending*, so a human approves them before they are studied. Each call logs its duration and token usage.

### Known limitations

- AI-generated answers can be wrong. That is why every question needs approval first.
- Scanned PDFs without text can't be read (no OCR).
- The web UI has no login. It only listens on `127.0.0.1`; use an SSH tunnel (see Deployment) instead of exposing it.
- Telegram only responds to the chat ID in `.env`.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests cover the SM-2 math, text chunking, AI-reply cleaning, database upgrades and the Telegram chat lock. They don't need an API key.

### Evals

`evals/runEvals.py` runs the real prompts on fixed sample material and checks the questions (count, types, key facts, no duplicates) and the topic name. It uses your real API key:

```bash
python evals/runEvals.py
```

Run it after changing a prompt or the model.

## Code style

The code deliberately uses camelCase everywhere, including Python, for readability across the Python and JavaScript parts. This breaks the usual Python style guide (PEP 8) on purpose. Run `ruff check .` for linting.

## License

[MIT](LICENSE)
