# StudyBot

A self-hosted spaced repetition system that turns dead time into study time. Upload source material, AI generates quiz questions, you vet them, and a Telegram bot delivers them to your phone on a scientifically-timed schedule.

**The loop:** Source material → AI-generated questions → human vetting → spaced repetition → Telegram push notifications → self-grading → interval adjustment → repeat.

## How It Works

1. **Upload** lecture notes, PDFs, or paste text through the web UI
2. **AI generates** quiz questions using Groq (Llama 3.3 70B) — free tier
3. **Review and approve** questions before they enter rotation
4. **Study via Telegram** — the bot sends due questions on schedule, you grade yourself, and the SM-2 algorithm adjusts intervals

Everything runs as a single process. The Telegram bot is embedded in the FastAPI app, sharing the same database. Deploy it on a server and your phone gets study notifications 24/7 — no laptop required.

## Tech Stack

- **Backend:** Python, FastAPI, SQLite
- **Frontend:** Vanilla HTML/CSS/JS in `static/` (no build tools)
- **AI:** Groq API (free tier), Llama 3.3 70B
- **Notifications:** python-telegram-bot
- **PDF extraction:** PyMuPDF
- **Scheduling:** SM-2 spaced repetition algorithm

## Setup

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/StudyBot.git
cd StudyBot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your keys:

- **`GROQ_API_KEY`** — Get one free at [console.groq.com](https://console.groq.com)
- **`TELEGRAM_BOT_TOKEN`** — Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
- **`TELEGRAM_CHAT_ID`** — Send a message to your bot, then visit `https://api.telegram.org/bot<your-token>/getUpdates` to find your chat ID

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

## License

[MIT](LICENSE)
