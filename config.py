import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("studyBot")


groqApiKey = os.getenv("GROQ_API_KEY", "").strip()
if not groqApiKey or groqApiKey == "your_key_here":
    print("ERROR: Set a valid GROQ_API_KEY in .env")
    sys.exit(1)

groqModel = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

telegramBotToken = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
telegramChatId = os.getenv("TELEGRAM_CHAT_ID", "").strip()

nudgeIntervalSeconds = 3 * 60 * 60
studyHoursStart = 8
studyHoursEnd = 23

databasePath = "spaced_rep.db"

maxInputCharacters = 100_000
maxPdfBytes = 20 * 1024 * 1024
maxPdfPages = 200
maxJsonBodyBytes = 512 * 1024

allowedQuestionTypes = ["factual", "conceptual", "application"]
allowedStatuses = ["pending", "approved"]
