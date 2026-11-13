import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TELEGRAM_CHAT_ID", "12345")
sys.path.insert(0, str(Path(__file__).parent.parent))
