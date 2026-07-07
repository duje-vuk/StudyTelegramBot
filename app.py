import os
import sys
import json
import time
import asyncio
import logging
import textwrap
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
import fitz  # pymupdf

# Telegram imports (graceful — bot features disabled if not installed)
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        Application,
        CommandHandler,
        CallbackQueryHandler,
        ContextTypes,
    )
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

load_dotenv()

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Telegram Bot Config ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_ENABLED = bool(TELEGRAM_AVAILABLE and BOT_TOKEN and BOT_TOKEN != "your_token_here" and CHAT_ID)
POLL_INTERVAL = 3 * 60 * 60  # Nudge every 3 hours
STUDY_HOURS_START = 8    # Don't notify before 8 AM
STUDY_HOURS_END = 23     # Don't notify after 11 PM

if TELEGRAM_AVAILABLE and not TELEGRAM_ENABLED:
    logger.warning("Telegram bot disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")

# --- Fail fast if no API key ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY or GROQ_API_KEY == "your_key_here":
    print("ERROR: Set a valid GROQ_API_KEY in .env")
    sys.exit(1)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Replaces deprecated @app.on_event("startup"/"shutdown")
    await _start_telegram_bot()
    yield
    await _stop_telegram_bot()


app = FastAPI(lifespan=lifespan)
client = Groq(api_key=GROQ_API_KEY)

# --- Database ---
DB_PATH = "spaced_rep.db"


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL is persistent (set once in init_db); only per-connection pragmas here.
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = get_db()
    conn.execute("PRAGMA journal_mode=WAL")  # persistent, set once
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'text',
            char_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'unknown',
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE SET NULL
        );

        CREATE INDEX IF NOT EXISTS idx_questions_status ON questions(status);
        CREATE INDEX IF NOT EXISTS idx_questions_source ON questions(source_id);
    """)

    # Stage 3 migration: add SM-2 columns if they don't exist
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(questions)").fetchall()}
    sm2_columns = {
        "easiness_factor": "REAL NOT NULL DEFAULT 2.5",
        "interval": "INTEGER NOT NULL DEFAULT 0",
        "repetitions": "INTEGER NOT NULL DEFAULT 0",
        "next_review": "TEXT",  # ISO 8601 UTC, NULL = not yet scheduled
        "paused": "INTEGER NOT NULL DEFAULT 0",  # 1 = schedule paused
    }
    for col, col_type in sm2_columns.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE questions ADD COLUMN {col} {col_type}")

    # Migration: add topic_name to sources if it doesn't exist
    source_cols = {row[1] for row in conn.execute("PRAGMA table_info(sources)").fetchall()}
    if "topic_name" not in source_cols:
        conn.execute("ALTER TABLE sources ADD COLUMN topic_name TEXT")

    # Index for due-question lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_next_review ON questions(next_review)")
    # Composite index matching the due-question WHERE clause exactly
    conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_due ON questions(status, paused, next_review)")
    conn.commit()
    conn.close()


init_db()

# --- Groq config ---
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a study question generator. Given source material, generate high-quality
    quiz questions for spaced repetition review.

    Rules:
    - Generate 5-10 questions per chunk of material
    - Focus on facts, concepts, and relationships worth remembering long-term
    - Avoid trivial, overly obvious, or ambiguous questions
    - Each question should have a clear, concise correct answer
    - Mix question types: factual recall, conceptual understanding, and application

    Respond with ONLY a JSON object in this exact format:
    {"questions": [{"question": "...", "answer": "...", "type": "factual|conceptual|application"}, ...]}

    No markdown, no preamble, no explanation. Just the JSON object.
""")

CHUNK_SIZE = 8000
CHUNK_OVERLAP = 500
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_CONCURRENCY = 4  # parallel chunk calls; 429s handled by exponential backoff

MAX_INPUT_CHARS = 100_000
MAX_CHUNKS = 15
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200


def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, n)
        chunks.append(text[start:end])
        if end >= n:
            break  # don't emit a trailing overlap-only chunk (wasted API call)
        start = end - CHUNK_OVERLAP
    return chunks


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pymupdf. Capped to prevent memory bombs."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count > MAX_PDF_PAGES:
        doc.close()
        raise ValueError(f"PDF has {doc.page_count} pages. Max is {MAX_PDF_PAGES}.")
    pages = []
    total_chars = 0
    for page in doc:
        text = page.get_text()
        total_chars += len(text)
        if total_chars > MAX_INPUT_CHARS * 2:
            # Decompressed text is way too large — likely a crafted PDF
            pages.append(text[:MAX_INPUT_CHARS * 2 - (total_chars - len(text))])
            break
        pages.append(text)
    doc.close()
    return "\n".join(pages)


def validate_questions(raw_parsed: dict) -> list[dict]:
    questions = raw_parsed.get("questions")
    if not isinstance(questions, list):
        return []
    valid = []
    allowed_types = {"factual", "conceptual", "application"}
    for item in questions:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        t = item.get("type", "unknown")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        if not q.strip() or not a.strip():
            continue
        if t not in allowed_types:
            t = "unknown"
        valid.append({
            "question": q.strip()[:1000],
            "answer": a.strip()[:2000],
            "type": t,
        })
    return valid


def call_groq(text_chunk: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate study questions from this material:\n\n{text_chunk}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            raw = response.choices[0].message.content
            logger.debug("Groq raw response: %s", raw[:500])
            parsed = json.loads(raw)
            return validate_questions(parsed)
        except json.JSONDecodeError:
            logger.warning("Groq returned invalid JSON on attempt %d", attempt + 1)
            if attempt == retries - 1:
                return []
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = (2 ** attempt) * 2
                logger.info("Rate limited, waiting %ds...", wait)
                time.sleep(wait)  # runs in a worker thread, doesn't block event loop
            else:
                # Sanitize: don't log full exception which may contain API key fragments
                err_type = type(e).__name__
                err_msg = str(e)[:200]
                logger.warning("Groq API error (attempt %d): [%s] %s", attempt + 1, err_type, err_msg)
                if attempt == retries - 1:
                    return []
    return []


def generate_topic_name(text_sample: str) -> str:
    """Use Groq to generate a short topic name from source material."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Given source material, respond with ONLY a short topic name (2-5 words) that describes the subject. Examples: 'Digital Logic', 'Cell Biology', 'French Revolution', 'Linear Algebra'. No quotes, no punctuation, no explanation."},
                {"role": "user", "content": text_sample[:3000]},
            ],
            temperature=0.1,
        )
        name = response.choices[0].message.content.strip().strip('"\'.')
        return name[:100] if name else "Unknown Topic"
    except Exception as e:
        logger.warning(f"Failed to generate topic name: {e}")
        return "Unknown Topic"


async def generate_from_text(text: str, source_title: str, source_type: str = "text") -> dict:
    """Chunk text, call Groq concurrently, save source + pending questions to DB."""
    chunks = chunk_text(text)
    sem = asyncio.Semaphore(GROQ_CONCURRENCY)

    async def process_chunk(chunk: str) -> list[dict]:
        async with sem:
            return await asyncio.to_thread(call_groq, chunk)

    # Topic name + all chunk calls run concurrently (bounded by semaphore)
    topic_task = asyncio.create_task(asyncio.to_thread(generate_topic_name, text))
    results = await asyncio.gather(*(process_chunk(c) for c in chunks))
    topic_name = await topic_task
    all_questions = [q for chunk_qs in results for q in chunk_qs]

    # Save to database
    now = datetime.now(timezone.utc).isoformat()
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO sources (title, type, char_count, created_at, topic_name) VALUES (?, ?, ?, ?, ?)",
            (source_title, source_type, len(text), now, topic_name)
        )
        source_id = cursor.lastrowid

        conn.executemany(
            "INSERT INTO questions (source_id, question, answer, type, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
            [(source_id, q["question"], q["answer"], q["type"], now) for q in all_questions]
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "questions": all_questions,
        "chunk_count": len(chunks),
        "source_id": source_id,
        "topic_name": topic_name,
    }


# --- SM-2 Algorithm ---


MAX_INTERVAL_DAYS = 365  # Let SM-2 drive intervals naturally


def sm2(quality: int, easiness_factor: float, interval: int, repetitions: int) -> tuple[float, int, int]:
    """
    SM-2 spaced repetition algorithm with interval cap.
    quality: 0-5 (0-2 = fail, 3-5 = pass)
    Returns: (new_easiness_factor, new_interval_days, new_repetitions)
    """
    # Clamp quality to 0-5
    quality = max(0, min(5, quality))

    # Update easiness factor
    new_ef = easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, new_ef)

    if quality < 3:
        # Failed — reset
        return new_ef, 1, 0

    # Passed
    new_reps = repetitions + 1
    if new_reps == 1:
        new_interval = 1
    elif new_reps == 2:
        new_interval = 3
    else:
        new_interval = round(interval * new_ef)

    # Cap interval
    new_interval = min(new_interval, MAX_INTERVAL_DAYS)

    return new_ef, new_interval, new_reps


def schedule_question(conn, question_id: int, quality: int):
    """Apply SM-2 to a question and update its next_review date."""
    row = conn.execute(
        "SELECT easiness_factor, interval, repetitions FROM questions WHERE id = ?",
        (question_id,)
    ).fetchone()
    if not row:
        return None

    ef, interval, reps = row["easiness_factor"], row["interval"], row["repetitions"]
    new_ef, new_interval, new_reps = sm2(quality, ef, interval, reps)

    next_review = (datetime.now(timezone.utc) + timedelta(days=new_interval)).isoformat()

    conn.execute(
        "UPDATE questions SET easiness_factor = ?, interval = ?, repetitions = ?, next_review = ? WHERE id = ?",
        (new_ef, new_interval, new_reps, next_review, question_id)
    )
    conn.commit()
    return {
        "easiness_factor": new_ef,
        "interval": new_interval,
        "repetitions": new_reps,
        "next_review": next_review,
    }


# --- API Routes ---

MAX_JSON_BODY = 512 * 1024  # 512KB max for JSON API endpoints


async def read_json_body(request: Request):
    """Shared size-capped JSON body parser. Returns (dict, None) or (None, error response)."""
    body_bytes = await request.body()
    if len(body_bytes) > MAX_JSON_BODY:
        return None, JSONResponse({"error": "Request body too large"}, status_code=413)
    try:
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        return None, JSONResponse({"error": "Invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return None, JSONResponse({"error": "JSON body must be an object"}, status_code=400)
    return body, None


@app.post("/generate")
async def generate_questions(request: Request):
    body, err = await read_json_body(request)
    if err:
        return err

    text = body.get("text", "")
    title = body.get("title", "")
    if not isinstance(title, str):
        title = ""
    title = title.strip() or "Untitled"
    if not isinstance(text, str):
        return JSONResponse({"error": "text must be a string"}, status_code=400)

    text = text.strip()
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if len(text) > MAX_INPUT_CHARS:
        return JSONResponse(
            {"error": f"Input too long ({len(text)} chars). Max is {MAX_INPUT_CHARS}."},
            status_code=413,
        )

    result = await generate_from_text(text, title[:200], "text")
    return result


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return JSONResponse({"error": "File must be a PDF"}, status_code=400)

    # Stream-read with a hard cap instead of loading unbounded bytes into memory
    buf = bytearray()
    while True:
        part = await file.read(1024 * 1024)
        if not part:
            break
        buf.extend(part)
        if len(buf) > MAX_PDF_BYTES:
            return JSONResponse({"error": "PDF too large. Max is 20MB."}, status_code=413)
    pdf_bytes = bytes(buf)

    try:
        # PDF parsing is CPU-bound — run off the event loop so the server stays responsive
        text = await asyncio.to_thread(extract_pdf_text, pdf_bytes)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=413)
    except Exception:
        return JSONResponse({"error": "Failed to read PDF. File may be corrupted or encrypted."}, status_code=400)

    text = text.strip()
    if not text:
        return JSONResponse({"error": "No text found in PDF. It may be scanned images without OCR."}, status_code=400)

    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]

    title = file.filename or "Uploaded PDF"
    # Sanitize: strip path components, control chars, limit length
    title = title.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    title = "".join(c for c in title if c.isprintable())
    title = title.strip() or "Uploaded PDF"
    result = await generate_from_text(text, title[:200], "pdf")
    result["extracted_chars"] = len(text)
    return result


# --- Question CRUD ---

@app.get("/api/questions")
def list_questions(status: str = None, source_id: int = None):
    # sync def → FastAPI runs it in a worker thread; blocking sqlite never stalls the event loop
    conn = get_db()
    try:
        query = "SELECT q.*, s.title as source_title FROM questions q LEFT JOIN sources s ON q.source_id = s.id WHERE 1=1"
        params = []
        if status:
            if status not in ("pending", "approved"):
                return JSONResponse({"error": "Invalid status filter"}, status_code=400)
            query += " AND q.status = ?"
            params.append(status)
        if source_id is not None:
            query += " AND q.source_id = ?"
            params.append(source_id)
        query += " ORDER BY q.created_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.put("/api/questions/{question_id}")
async def update_question(question_id: int, request: Request):
    body, err = await read_json_body(request)
    if err:
        return err

    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM questions WHERE id = ?", (question_id,)).fetchone()
        if not existing:
            return JSONResponse({"error": "Question not found"}, status_code=404)

        # Whitelist of allowed fields — never construct column names from user input
        ALLOWED_FIELDS = {"question", "answer", "type", "status"}
        updates = []
        params = []
        for field in ("question", "answer", "type", "status"):
            if field not in body or field not in ALLOWED_FIELDS:
                continue
            val = str(body[field]).strip()
            if field == "question":
                val = val[:1000]
            elif field == "answer":
                val = val[:2000]
            elif field == "status" and val not in ("pending", "approved"):
                continue
            elif field == "type" and val not in ("factual", "conceptual", "application", "unknown"):
                continue
            updates.append(f"{field} = ?")
            params.append(val)

        if not updates:
            return JSONResponse({"error": "No valid fields to update"}, status_code=400)

        # If approving, set next_review so it enters study rotation
        if body.get("status") == "approved":
            updates.append("next_review = COALESCE(next_review, ?)")
            params.append(datetime.now(timezone.utc).isoformat())

        params.append(question_id)
        conn.execute(f"UPDATE questions SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

        row = conn.execute(
            "SELECT q.*, s.title as source_title FROM questions q LEFT JOIN sources s ON q.source_id = s.id WHERE q.id = ?",
            (question_id,)
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.delete("/api/questions/{question_id}")
def delete_question(question_id: int):
    conn = get_db()
    try:
        existing = conn.execute("SELECT id FROM questions WHERE id = ?", (question_id,)).fetchone()
        if not existing:
            return JSONResponse({"error": "Question not found"}, status_code=404)
        conn.execute("DELETE FROM questions WHERE id = ?", (question_id,))
        conn.commit()
        return {"deleted": question_id}
    finally:
        conn.close()


@app.post("/api/questions/batch")
async def batch_update(request: Request):
    """Batch approve or delete questions."""
    body, err = await read_json_body(request)
    if err:
        return err

    ids = body.get("ids", [])
    action = body.get("action", "")

    if not isinstance(ids, list) or not ids:
        return JSONResponse({"error": "ids must be a non-empty list"}, status_code=400)
    if len(ids) > 500:
        return JSONResponse({"error": "Too many ids. Max is 500 per batch."}, status_code=400)
    if action not in ("approve", "delete"):
        return JSONResponse({"error": "action must be 'approve' or 'delete'"}, status_code=400)

    try:
        ids = [int(i) for i in ids]
    except (ValueError, TypeError):
        return JSONResponse({"error": "ids must be integers"}, status_code=400)

    conn = get_db()
    try:
        placeholders = ",".join("?" * len(ids))
        if action == "approve":
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(f"UPDATE questions SET status = 'approved', next_review = COALESCE(next_review, ?) WHERE id IN ({placeholders})", [now] + ids)
        elif action == "delete":
            conn.execute(f"DELETE FROM questions WHERE id IN ({placeholders})", ids)
        conn.commit()
        return {"action": action, "count": len(ids)}
    finally:
        conn.close()


@app.get("/api/sources")
def list_sources():
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT s.*,
                   COUNT(q.id) as total_questions,
                   SUM(CASE WHEN q.status = 'approved' THEN 1 ELSE 0 END) as approved_count,
                   SUM(CASE WHEN q.status = 'pending' THEN 1 ELSE 0 END) as pending_count
            FROM sources s
            LEFT JOIN questions q ON q.source_id = s.id
            GROUP BY s.id
            ORDER BY s.created_at DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# --- Stage 3: SM-2 Scheduling Endpoints ---


@app.get("/api/due")
def get_due_questions(limit: int = 20):
    """Get questions that are due for study."""
    limit = max(1, min(limit, 100))  # clamp: negative LIMIT in sqlite means "no limit"
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute("""
            SELECT q.*, s.title as source_title
            FROM questions q
            LEFT JOIN sources s ON q.source_id = s.id
            WHERE q.status = 'approved'
              AND q.paused = 0
              AND (q.next_review IS NULL OR q.next_review <= ?)
            ORDER BY
                q.next_review IS NULL DESC,
                q.next_review ASC
            LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/review")
async def submit_review(request: Request):
    """Submit a self-grade for a question. Updates SM-2 fields."""
    body, err = await read_json_body(request)
    if err:
        return err

    question_id = body.get("question_id")
    quality = body.get("quality")

    if not isinstance(question_id, int):
        return JSONResponse({"error": "question_id must be an integer"}, status_code=400)
    if not isinstance(quality, int) or quality < 0 or quality > 5:
        return JSONResponse({"error": "quality must be an integer 0-5"}, status_code=400)

    conn = get_db()
    try:
        existing = conn.execute(
            "SELECT id, status FROM questions WHERE id = ?", (question_id,)
        ).fetchone()
        if not existing:
            return JSONResponse({"error": "Question not found"}, status_code=404)
        if existing["status"] != "approved":
            return JSONResponse({"error": "Only approved questions can be reviewed"}, status_code=400)

        result = schedule_question(conn, question_id, quality)
        if not result:
            return JSONResponse({"error": "Failed to update"}, status_code=500)

        return {"question_id": question_id, "quality": quality, **result}
    finally:
        conn.close()


@app.get("/api/stats")
def get_study_stats():
    """Dashboard stats: due count, total in rotation, review history. Single table scan."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        week_later = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()

        row = conn.execute("""
            SELECT
                SUM(CASE WHEN paused = 0 AND (next_review IS NULL OR next_review <= :now) THEN 1 ELSE 0 END) AS due_now,
                COUNT(*) AS total_approved,
                SUM(CASE WHEN repetitions > 0 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN paused = 0 AND next_review > :now AND next_review <= :week THEN 1 ELSE 0 END) AS upcoming_7d,
                SUM(CASE WHEN paused = 1 THEN 1 ELSE 0 END) AS paused_count
            FROM questions
            WHERE status = 'approved'
        """, {"now": now, "week": week_later}).fetchone()

        return {
            "due_now": row["due_now"] or 0,
            "total_in_rotation": row["total_approved"] or 0,
            "reviewed_at_least_once": row["reviewed"] or 0,
            "upcoming_7_days": row["upcoming_7d"] or 0,
            "paused": row["paused_count"] or 0,
        }
    finally:
        conn.close()


@app.get("/api/counts")
def get_counts():
    """Lightweight badge counts — replaces fetching the entire question table client-side."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        row = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS approved,
                SUM(CASE WHEN status = 'approved' AND paused = 0
                         AND (next_review IS NULL OR next_review <= ?) THEN 1 ELSE 0 END) AS due
            FROM questions
        """, (now,)).fetchone()
        return {
            "total": row["total"] or 0,
            "pending": row["pending"] or 0,
            "approved": row["approved"] or 0,
            "due": row["due"] or 0,
        }
    finally:
        conn.close()


@app.get("/api/upcoming")
def get_upcoming_reviews():
    """Get approved questions with future review dates, grouped by day."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        # Only the columns the UI renders — answers/EF/reps were dead payload
        rows = conn.execute("""
            SELECT q.id, q.question, q.next_review, q.interval,
                   s.title as source_title
            FROM questions q
            LEFT JOIN sources s ON q.source_id = s.id
            WHERE q.status = 'approved'
              AND q.next_review IS NOT NULL
              AND q.next_review > ?
            ORDER BY q.next_review ASC
            LIMIT 500
        """, (now,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/api/pause")
def pause_schedule():
    """Mute all approved questions — nothing will be due until unmuted."""
    conn = get_db()
    try:
        result = conn.execute(
            "UPDATE questions SET paused = 1, next_review = NULL WHERE status = 'approved' AND paused = 0"
        )
        conn.commit()
        return {"paused": result.rowcount}
    finally:
        conn.close()


@app.post("/api/resume")
def resume_schedule():
    """Unmute all paused questions — sets them all to due now."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        result = conn.execute(
            "UPDATE questions SET paused = 0, next_review = ? WHERE status = 'approved' AND paused = 1",
            (now,)
        )
        conn.commit()
        return {"resumed": result.rowcount}
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# =============================================
# TELEGRAM BOT (embedded — runs in background)
# =============================================

_telegram_app = None  # Holds the running Application instance
_active_sessions = {}  # chat_id -> {"queue": [question_ids], "requeue": [question_ids]}




def _esc(text: str) -> str:
    """Escape HTML special characters for Telegram messages."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def _send_question(context: ContextTypes.DEFAULT_TYPE, q: dict, chat_id: str = None, remaining: int = None):
    """Send a question as a Telegram message with 'Show Answer' button."""
    target = chat_id or CHAT_ID
    text = f"<b>Question:</b>\n{_esc(q['question'])}"
    if remaining is not None:
        text += f"\n\n<i>({remaining} remaining)</i>"
    keyboard = [[InlineKeyboardButton("Show Answer", callback_data=f"show_{q['id']}")]]
    await context.bot.send_message(
        chat_id=target,
        text=text,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def _get_due_for_bot(limit=1):
    """Get approved, unpaused questions that are due now."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute("""
            SELECT q.id, q.question, q.answer, q.type,
                   q.easiness_factor, q.interval, q.repetitions,
                   s.title as source_title
            FROM questions q
            LEFT JOIN sources s ON q.source_id = s.id
            WHERE q.status = 'approved'
              AND q.paused = 0
              AND (q.next_review IS NULL OR q.next_review <= ?)
            ORDER BY q.next_review IS NULL DESC, q.next_review ASC
            LIMIT ?
        """, (now, limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _get_due_count():
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        return conn.execute("""
            SELECT COUNT(*) FROM questions
            WHERE status = 'approved' AND paused = 0
              AND (next_review IS NULL OR next_review <= ?)
        """, (now,)).fetchone()[0]
    finally:
        conn.close()


def _grade_question_bot(question_id, quality):
    """Apply SM-2 grade via bot. Returns new interval or None."""
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT easiness_factor, interval, repetitions, status FROM questions WHERE id = ?",
            (question_id,)
        ).fetchone()
        if not row or row["status"] != "approved":
            return None
        new_ef, new_iv, new_reps = sm2(quality, row["easiness_factor"], row["interval"], row["repetitions"])
        next_review = (datetime.now(timezone.utc) + timedelta(days=new_iv)).isoformat()
        conn.execute(
            "UPDATE questions SET easiness_factor = ?, interval = ?, repetitions = ?, next_review = ? WHERE id = ?",
            (new_ef, new_iv, new_reps, next_review, question_id)
        )
        conn.commit()
        return new_iv
    finally:
        conn.close()


# --- Telegram command handlers ---

async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Study Bot active.\n\n"
        "Commands:\n"
        "/study — Start a study session\n"
        "/pause — Stop current session\n"
        "/reset — Reset all progress, start fresh\n"
        "/due — Check how many questions are due\n"
        "/stats — View study stats\n"
        "/mute — Silence all notifications\n"
        "/unmute — Resume notifications\n\n"
        "I'll nudge you every few hours when questions are due."
    )


async def _cmd_study(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    # If already in a session, just send next question
    if _active_sessions.get(chat_id):
        await _send_next_in_session(context, chat_id)
        return
    # Start a new session: load all due question IDs
    questions = _get_due_for_bot(limit=200)
    if not questions:
        await update.message.reply_text("No questions due right now. You're caught up!")
        return
    REQUEUE_COUNTS = {5: 1, 3: 2, 1: 3}  # Easy=1, Revise=2, Don't Know=3
    _active_sessions[chat_id] = {
        "queue": [q["id"] for q in questions],
        "requeue": [],
        "total": len(questions),
        "answered": 0,
        "requeue_counts": REQUEUE_COUNTS,
        "question_state": {},  # question_id -> {"requeues_left": int, "last_quality": int}
    }
    await update.message.reply_text(f"Starting study session — {len(questions)} question{'s' if len(questions) != 1 else ''} due. /pause to stop.")
    await _send_next_in_session(context, chat_id)


async def _send_next_in_session(context: ContextTypes.DEFAULT_TYPE, chat_id: str):
    """Send the next question from the session queue."""
    session = _active_sessions.get(chat_id)
    if not session:
        return
    # If main queue is empty, move requeue items in
    if not session["queue"] and session["requeue"]:
        session["queue"] = session["requeue"]
        session["requeue"] = []
    if not session["queue"]:
        # Session complete
        total = session["total"]
        del _active_sessions[chat_id]
        await context.bot.send_message(chat_id=chat_id, text=f"Session complete! Reviewed {total} question{'s' if total != 1 else ''}. You're caught up!")
        return
    # Loop (not recursion) past deleted questions — avoids deep recursion on stale queues
    row = None
    conn = get_db()
    try:
        while session["queue"] or session["requeue"]:
            if not session["queue"]:
                session["queue"] = session["requeue"]
                session["requeue"] = []
            qid = session["queue"].pop(0)
            row = conn.execute("""
                SELECT q.id, q.question, q.answer, q.type,
                       q.easiness_factor, q.interval, q.repetitions
                FROM questions q WHERE q.id = ?
            """, (qid,)).fetchone()
            if row:
                break
    finally:
        conn.close()
    if not row:
        total = session["total"]
        del _active_sessions[chat_id]
        await context.bot.send_message(chat_id=chat_id, text=f"Session complete! Reviewed {total} question{'s' if total != 1 else ''}. You're caught up!")
        return
    remaining = len(session["queue"]) + len(session["requeue"])
    await _send_question(context, dict(row), chat_id=chat_id, remaining=remaining)


async def _cmd_due(update: Update, context: ContextTypes.DEFAULT_TYPE):
    count = _get_due_count()
    if count == 0:
        await update.message.reply_text("No questions due. You're caught up.")
    else:
        await update.message.reply_text(f"{count} question{'s' if count != 1 else ''} due.")


async def _cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        due = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE status = 'approved' AND paused = 0 AND (next_review IS NULL OR next_review <= ?)",
            (now,)
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved'").fetchone()[0]
        reviewed = conn.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved' AND repetitions > 0").fetchone()[0]
        paused = conn.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved' AND paused = 1").fetchone()[0]
    finally:
        conn.close()
    await update.message.reply_text(f"Due now: {due}\nIn rotation: {total}\nReviewed at least once: {reviewed}\nPaused: {paused}")


async def _cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Reset all SM-2 progress and restart the current session from scratch."""
    chat_id = str(update.effective_chat.id)
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        result = conn.execute("""
            UPDATE questions SET easiness_factor = 2.5, interval = 0, repetitions = 0, next_review = ?
            WHERE status = 'approved' AND paused = 0
        """, (now,))
        conn.commit()
        count = result.rowcount
    finally:
        conn.close()
    # End current session so /study starts fresh
    _active_sessions.pop(chat_id, None)
    await update.message.reply_text(f"Reset {count} questions to fresh. All due now — /study to start.")


async def _cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stop the current study session."""
    chat_id = str(update.effective_chat.id)
    session = _active_sessions.pop(chat_id, None)
    if session:
        answered = session["answered"]
        await update.message.reply_text(f"Study paused. Answered {answered} question{'s' if answered != 1 else ''} this session.")
    else:
        await update.message.reply_text("No active study session.")


async def _cmd_mute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Mute all notifications — no nudges, no scheduled sends."""
    conn = get_db()
    try:
        result = conn.execute(
            "UPDATE questions SET paused = 1, next_review = NULL WHERE status = 'approved' AND paused = 0"
        )
        conn.commit()
        count = result.rowcount
    finally:
        conn.close()
    # Also end any active session
    chat_id = str(update.effective_chat.id)
    _active_sessions.pop(chat_id, None)
    await update.message.reply_text(f"Muted {count} questions. No notifications until you /unmute.")


async def _cmd_unmute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Unmute — resume all questions, set them due now."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        result = conn.execute(
            "UPDATE questions SET paused = 0, next_review = ? WHERE status = 'approved' AND paused = 1",
            (now,)
        )
        conn.commit()
        count = result.rowcount
    finally:
        conn.close()
    await update.message.reply_text(f"Unmuted {count} questions. They're all due now — /study to start.")



async def _handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button presses (show answer, grade)."""
    query = update.callback_query
    await query.answer()
    data = query.data
    chat_id = str(query.message.chat_id)

    if data.startswith("show_"):
        question_id = int(data.split("_")[1])
        conn = get_db()
        try:
            row = conn.execute("SELECT question, answer, type FROM questions WHERE id = ?", (question_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            await query.edit_message_text("Question not found (may have been deleted).")
            return
        text = f"<b>Question:</b>\n{_esc(row['question'])}\n\n<b>Answer:</b>\n{_esc(row['answer'])}"
        keyboard = [[
            InlineKeyboardButton("✅ Easy", callback_data=f"grade_{question_id}_5"),
            InlineKeyboardButton("🔄 Revise", callback_data=f"grade_{question_id}_3"),
            InlineKeyboardButton("❌ Don't Know Yet", callback_data=f"grade_{question_id}_1"),
        ]]
        await query.edit_message_text(text=text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith("grade_"):
        parts = data.split("_")
        question_id = int(parts[1])
        quality = int(parts[2])
        grade_labels = {1: "Don't Know Yet", 3: "Revise", 5: "Easy"}
        label = grade_labels.get(quality, str(quality))

        session = _active_sessions.get(chat_id)

        if session:
            state = session["question_state"]
            if question_id not in state:
                # First time seeing this question — set requeue count
                requeues = session["requeue_counts"].get(quality, 1)
                state[question_id] = {"requeues_left": requeues, "last_quality": quality}
            else:
                # Subsequent visit — update last grade, decrement
                state[question_id]["requeues_left"] -= 1
                state[question_id]["last_quality"] = quality

            qs = state[question_id]
            session["answered"] += 1

            if qs["requeues_left"] > 0:
                # Still has requeues — put it back
                session["requeue"].append(question_id)
                await query.edit_message_text(
                    f"<b>{label}</b> — will come back {qs['requeues_left']} more time{'s' if qs['requeues_left'] != 1 else ''}.",
                    parse_mode="HTML",
                )
                await _send_next_in_session(context, chat_id)
            else:
                # Final showing — apply SM-2 with last grade
                final_quality = qs["last_quality"]
                new_interval = _grade_question_bot(question_id, final_quality)
                if new_interval is None:
                    await query.edit_message_text("Failed to grade — question may have been deleted.")
                    return
                final_label = grade_labels.get(final_quality, str(final_quality))
                await query.edit_message_text(
                    f"<b>{final_label}</b> — next review in {new_interval} day{'s' if new_interval != 1 else ''}.",
                    parse_mode="HTML",
                )
                del state[question_id]
                await _send_next_in_session(context, chat_id)
        else:
            # No active session — just grade via SM-2 directly
            new_interval = _grade_question_bot(question_id, quality)
            if new_interval is None:
                await query.edit_message_text("Failed to grade — question may have been deleted.")
                return
            await query.edit_message_text(
                f"<b>{label}</b> — next review in {new_interval} day{'s' if new_interval != 1 else ''}.",
                parse_mode="HTML",
            )


def _get_due_by_topic():
    """Get due question counts grouped by topic name."""
    conn = get_db()
    try:
        now = datetime.now(timezone.utc).isoformat()
        rows = conn.execute("""
            SELECT COALESCE(s.topic_name, s.title, 'Unknown') as topic, COUNT(*) as cnt
            FROM questions q
            LEFT JOIN sources s ON q.source_id = s.id
            WHERE q.status = 'approved' AND q.paused = 0
              AND (q.next_review IS NULL OR q.next_review <= ?)
            GROUP BY topic
            ORDER BY cnt DESC
        """, (now,)).fetchall()
        return [(row["topic"], row["cnt"]) for row in rows]
    finally:
        conn.close()


async def _scheduled_check(context: ContextTypes.DEFAULT_TYPE):
    """Periodically nudge if there are due questions (doesn't send questions directly)."""
    # System local time (DST-aware) — was hardcoded CET, which drifted 1h every summer
    now = datetime.now().astimezone()
    if now.hour < STUDY_HOURS_START or now.hour >= STUDY_HOURS_END:
        logger.info(f"Outside study hours ({STUDY_HOURS_START}-{STUDY_HOURS_END}). Skipping.")
        return
    # Don't nudge if user is mid-session
    if _active_sessions.get(CHAT_ID):
        logger.info("User is in active study session. Skipping nudge.")
        return
    topic_counts = _get_due_by_topic()
    if not topic_counts:
        logger.info("No due questions.")
        return
    total = sum(cnt for _, cnt in topic_counts)
    lines = [f"You have {total} question{'s' if total != 1 else ''} due:"]
    for topic, cnt in topic_counts:
        lines.append(f"  • {topic}: {cnt}")
    lines.append("\n/study to start.")
    await context.bot.send_message(chat_id=CHAT_ID, text="\n".join(lines))
    logger.info(f"Sent nudge: {total} due across {len(topic_counts)} topics.")


# --- Telegram bot lifecycle (start/stop with FastAPI) ---

async def _start_telegram_bot():
    """Initialize and start the Telegram bot as a background task."""
    global _telegram_app
    if not TELEGRAM_ENABLED:
        logger.info("Telegram bot not enabled. Skipping.")
        return

    logger.info("Starting embedded Telegram bot...")
    _telegram_app = Application.builder().token(BOT_TOKEN).build()

    _telegram_app.add_handler(CommandHandler("start", _cmd_start))
    _telegram_app.add_handler(CommandHandler("study", _cmd_study))
    _telegram_app.add_handler(CommandHandler("due", _cmd_due))
    _telegram_app.add_handler(CommandHandler("stats", _cmd_stats))
    _telegram_app.add_handler(CommandHandler("pause", _cmd_pause))
    _telegram_app.add_handler(CommandHandler("reset", _cmd_reset))
    _telegram_app.add_handler(CommandHandler("mute", _cmd_mute))
    _telegram_app.add_handler(CommandHandler("unmute", _cmd_unmute))
    _telegram_app.add_handler(CallbackQueryHandler(_handle_callback))

    _telegram_app.job_queue.run_repeating(
        _scheduled_check,
        interval=POLL_INTERVAL,
        first=10,
    )

    # Initialize and start polling without blocking
    await _telegram_app.initialize()
    await _telegram_app.start()
    await _telegram_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram bot started.")


async def _stop_telegram_bot():
    """Gracefully shut down the Telegram bot."""
    global _telegram_app
    if _telegram_app is None:
        return
    logger.info("Stopping Telegram bot...")
    await _telegram_app.updater.stop()
    await _telegram_app.stop()
    await _telegram_app.shutdown()
    _telegram_app = None
    logger.info("Telegram bot stopped.")


# --- Frontend ---
HTML_PAGE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spaced Rep</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 2rem;
        }
        h1 {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #fff;
        }
        .container { max-width: 800px; margin: 0 auto; }

        /* --- Main nav --- */
        .main-nav {
            display: flex;
            gap: 0;
            margin-bottom: 2rem;
            border-bottom: 2px solid #2a2a2a;
        }
        .main-nav button {
            padding: 0.7rem 1.6rem;
            background: none;
            border: none;
            color: #555;
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: color 0.15s;
        }
        .main-nav button.active {
            color: #fff;
            border-bottom-color: #fff;
        }
        .main-nav button:hover:not(.active) { color: #999; }
        .main-nav .badge {
            display: inline-block;
            background: #333;
            color: #aaa;
            font-size: 0.7rem;
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
            margin-left: 0.4rem;
            vertical-align: middle;
        }
        .main-nav button.active .badge {
            background: #fff;
            color: #000;
        }
        .view { display: none; }
        .view.active { display: block; }

        /* --- Subtabs (text/pdf) --- */
        .subtabs {
            display: flex;
            gap: 0;
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #2a2a2a;
        }
        .subtab {
            padding: 0.5rem 1.2rem;
            background: none;
            border: none;
            color: #666;
            font-size: 0.85rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            margin-bottom: -1px;
        }
        .subtab.active {
            color: #fff;
            border-bottom-color: #fff;
        }
        .subtab-content { display: none; }
        .subtab-content.active { display: block; }

        /* --- Inputs --- */
        textarea {
            width: 100%;
            height: 200px;
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            color: #e0e0e0;
            padding: 1rem;
            font-size: 0.95rem;
            font-family: inherit;
            resize: vertical;
            outline: none;
        }
        textarea:focus { border-color: #444; }
        textarea::placeholder { color: #555; }

        input[type="text"] {
            width: 100%;
            padding: 0.6rem 0.9rem;
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 0.9rem;
            font-family: inherit;
            outline: none;
            margin-bottom: 0.8rem;
        }
        input[type="text"]:focus { border-color: #444; }
        input[type="text"]::placeholder { color: #555; }

        /* --- Drop zone --- */
        .drop-zone {
            width: 100%;
            height: 160px;
            border: 2px dashed #2a2a2a;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            cursor: pointer;
            transition: border-color 0.15s;
        }
        .drop-zone:hover, .drop-zone.drag-over { border-color: #555; }
        .drop-zone .label { color: #666; font-size: 0.95rem; }
        .drop-zone .sublabel { color: #444; font-size: 0.8rem; }
        .file-name { margin-top: 0.5rem; font-size: 0.85rem; color: #aaa; }

        /* --- Buttons --- */
        .btn {
            padding: 0.6rem 1.4rem;
            border: none;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s;
        }
        .btn-primary { background: #fff; color: #000; }
        .btn-primary:hover { background: #ddd; }
        .btn-primary:disabled { background: #333; color: #666; cursor: not-allowed; }

        .btn-approve { background: #1a7a3a; color: #fff; }
        .btn-approve:hover { background: #1e8e44; }
        .btn-delete { background: #6b1c1c; color: #fff; }
        .btn-delete:hover { background: #832222; }
        .btn-edit { background: #333; color: #ddd; }
        .btn-edit:hover { background: #444; }
        .btn-sm { padding: 0.35rem 0.8rem; font-size: 0.8rem; }
        .btn-ghost {
            background: none;
            border: 1px solid #333;
            color: #aaa;
        }
        .btn-ghost:hover { border-color: #555; color: #fff; }

        /* --- Batch bar --- */
        .batch-bar {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.8rem 1rem;
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .batch-bar .count { font-size: 0.85rem; color: #aaa; flex: 1; }

        /* --- Cards --- */
        .question-card {
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 1.2rem;
            margin-bottom: 0.8rem;
            transition: border-color 0.15s;
        }
        .question-card.selected { border-color: #555; }
        .question-card .card-header {
            display: flex;
            align-items: flex-start;
            gap: 0.8rem;
        }
        .question-card .card-header input[type="checkbox"] {
            margin-top: 0.25rem;
            accent-color: #fff;
            cursor: pointer;
            width: 16px;
            height: 16px;
            flex-shrink: 0;
        }
        .question-card .card-body { flex: 1; min-width: 0; }
        .question-card .q {
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.4rem;
            line-height: 1.4;
        }
        .question-card .a { color: #aaa; line-height: 1.4; margin-bottom: 0.5rem; }
        .question-card .meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }
        .tag {
            display: inline-block;
            font-size: 0.72rem;
            color: #666;
            background: #1a1a1a;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
        }
        .tag.approved { color: #3a9; background: #0f2a20; }
        .tag.pending { color: #c90; background: #2a2000; }
        .question-card .actions {
            display: flex;
            gap: 0.4rem;
            margin-top: 0.6rem;
        }

        /* --- Edit mode --- */
        .edit-field {
            width: 100%;
            padding: 0.5rem 0.7rem;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 0.9rem;
            font-family: inherit;
            margin-bottom: 0.4rem;
            outline: none;
        }
        .edit-field:focus { border-color: #555; }
        textarea.edit-field { height: 60px; resize: vertical; }

        /* --- Status & info --- */
        .status-msg {
            margin-top: 0.8rem;
            font-size: 0.85rem;
            color: #888;
        }
        .char-count {
            font-size: 0.8rem;
            color: #555;
            margin-top: 0.4rem;
        }
        .char-count.warn { color: #c44; }
        .empty-state {
            text-align: center;
            padding: 3rem 1rem;
            color: #555;
            font-size: 0.95rem;
        }

        /* --- Filter bar --- */
        .filter-bar {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 0.35rem 0.9rem;
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 20px;
            color: #888;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.15s;
        }
        .filter-btn.active {
            background: #fff;
            color: #000;
            border-color: #fff;
        }
        .filter-btn:hover:not(.active) { border-color: #555; color: #ccc; }

        /* --- Source label --- */
        .source-label {
            font-size: 0.72rem;
            color: #556;
        }

        /* --- Study session --- */
        .study-progress {
            text-align: center;
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 1.2rem;
        }
        .flashcard {
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 2.5rem 2rem;
            min-height: 220px;
            cursor: pointer;
            transition: border-color 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .flashcard:hover { border-color: #444; }
        .flashcard-face { width: 100%; text-align: center; }
        .flashcard-label {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: #555;
            margin-bottom: 1.2rem;
        }
        .flashcard-content {
            font-size: 1.15rem;
            line-height: 1.6;
            color: #e0e0e0;
            max-width: 600px;
            margin: 0 auto;
        }
        .flashcard-hint {
            font-size: 0.78rem;
            color: #444;
            margin-top: 1.5rem;
        }
        .flashcard-source {
            font-size: 0.75rem;
            color: #444;
            margin-top: 1.2rem;
        }
        .grade-bar {
            margin-top: 1.5rem;
            text-align: center;
        }
        .grade-label {
            font-size: 0.85rem;
            color: #888;
            margin-bottom: 0.8rem;
        }
        .grade-row {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
        }
        .grade-btn {
            flex: 1;
            max-width: 90px;
            padding: 0.7rem 0.3rem;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            background: #141414;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.25rem;
        }
        .grade-btn:hover { border-color: #555; }
        .grade-num { font-size: 1rem; font-weight: 700; }
        .grade-text { font-size: 0.7rem; }
        .grade-0 .grade-num { color: #e44; }
        .grade-0:hover { background: #2a1010; border-color: #e44; }
        .grade-1 .grade-num { color: #e77; }
        .grade-1:hover { background: #2a1515; border-color: #e77; }
        .grade-2 .grade-num { color: #ea5; }
        .grade-2:hover { background: #2a2010; border-color: #ea5; }
        .grade-3 .grade-num { color: #cc3; }
        .grade-3:hover { background: #252510; border-color: #cc3; }
        .grade-4 .grade-num { color: #5c5; }
        .grade-4:hover { background: #102a10; border-color: #5c5; }
        .grade-5 .grade-num { color: #3a9; }
        .grade-5:hover { background: #0f2a20; border-color: #3a9; }
        .grade-0 .grade-text { color: #a55; }
        .grade-1 .grade-text { color: #a66; }
        .grade-2 .grade-text { color: #a85; }
        .grade-3 .grade-text { color: #995; }
        .grade-4 .grade-text { color: #5a5; }
        .grade-5 .grade-text { color: #3a9; }

        /* --- Pause button --- */
        .btn-pause {
            padding: 0.5rem 1.2rem;
            border: 1px solid #2a2a2a;
            border-radius: 6px;
            background: #141414;
            color: #aaa;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }
        .btn-pause:hover { border-color: #555; color: #fff; }
        .btn-pause.paused {
            background: #c90;
            color: #000;
            border-color: #c90;
        }
        .btn-pause.paused:hover { background: #da0; }
        .pause-status {
            font-size: 0.8rem;
            color: #888;
            margin-left: 0.8rem;
        }

        /* --- Upcoming reviews --- */
        .upcoming-day {
            margin-bottom: 1.2rem;
        }
        .upcoming-day-header {
            font-size: 0.8rem;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid #1a1a1a;
        }
        .upcoming-item {
            padding: 0.6rem 0.8rem;
            background: #141414;
            border: 1px solid #1a1a1a;
            border-radius: 6px;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
        }
        .upcoming-q {
            font-size: 0.85rem;
            color: #ccc;
            flex: 1;
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .upcoming-meta {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            flex-shrink: 0;
        }
        .upcoming-interval {
            font-size: 0.72rem;
            color: #555;
            background: #1a1a1a;
            padding: 0.15rem 0.45rem;
            border-radius: 4px;
        }

        /* --- Summary stats --- */
        .stats-row {
            display: flex;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .stat-box {
            background: #141414;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            flex: 1;
            text-align: center;
        }
        .stat-box .num {
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
        }
        .stat-box .label {
            font-size: 0.75rem;
            color: #666;
            margin-top: 0.2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Spaced Rep</h1>

        <div class="main-nav">
            <button class="active" onclick="switchView('generate')">Generate</button>
            <button onclick="switchView('review')">Review <span class="badge" id="pendingBadge">0</span></button>
            <button onclick="switchView('study')">Study <span class="badge" id="dueBadge">0</span></button>
            <button onclick="switchView('bank')">Question Bank <span class="badge" id="approvedBadge">0</span></button>
        </div>

        <!-- ======= GENERATE VIEW ======= -->
        <div id="view-generate" class="view active">
            <input type="text" id="sourceTitle" placeholder="Source title (e.g. &quot;Digital Logic Lecture 3&quot;)" maxlength="200">

            <div class="subtabs">
                <button class="subtab active" onclick="switchSubtab('text')">Paste Text</button>
                <button class="subtab" onclick="switchSubtab('pdf')">Upload PDF</button>
            </div>

            <div id="subtab-text" class="subtab-content active">
                <textarea id="source" placeholder="Paste your study material here..." maxlength="100000"></textarea>
                <div class="char-count" id="charCount">0 / 100,000</div>
            </div>

            <div id="subtab-pdf" class="subtab-content">
                <div class="drop-zone" id="dropZone">
                    <div class="label">Drop a PDF here or click to browse</div>
                    <div class="sublabel">Max 20MB / 200 pages</div>
                </div>
                <input type="file" id="fileInput" accept=".pdf" hidden>
                <div class="file-name" id="fileName"></div>
            </div>

            <button class="btn btn-primary" id="genBtn" onclick="generate()" style="margin-top:1rem;">Generate Questions</button>
            <div class="status-msg" id="genStatus"></div>
        </div>

        <!-- ======= REVIEW VIEW ======= -->
        <div id="view-review" class="view">
            <div id="reviewBatchBar" class="batch-bar" style="display:none;">
                <input type="checkbox" id="selectAllPending" onchange="toggleSelectAll('pending')">
                <span class="count" id="reviewSelCount">0 selected</span>
                <button class="btn btn-approve btn-sm" onclick="batchAction('approve', 'pending')">Approve Selected</button>
                <button class="btn btn-delete btn-sm" onclick="batchAction('delete', 'pending')">Delete Selected</button>
            </div>
            <div id="reviewList"></div>
        </div>

        <!-- ======= STUDY VIEW ======= -->
        <div id="view-study" class="view">
            <div class="stats-row" id="studyStats"></div>


            <div id="pauseBar" style="display:none; margin-bottom:1.5rem;">
                <button class="btn btn-pause" id="pauseBtn" onclick="togglePause()">Pause Schedule</button>
                <span class="pause-status" id="pauseStatus"></span>
            </div>

            <div id="studyEmpty" class="empty-state" style="display:none;">
                No questions due right now. Check back later or approve more questions.
            </div>

            <div id="studySession" style="display:none;">
                <div class="study-progress">
                    <span id="studyProgress">0 / 0</span>
                </div>

                <div class="flashcard" id="flashcard" onclick="flipCard()">
                    <div class="flashcard-face flashcard-front" id="cardFront">
                        <div class="flashcard-label">QUESTION</div>
                        <div class="flashcard-content" id="cardQuestion"></div>
                        <div class="flashcard-hint">Click to reveal answer</div>
                    </div>
                    <div class="flashcard-face flashcard-back" id="cardBack" style="display:none;">
                        <div class="flashcard-label">ANSWER</div>
                        <div class="flashcard-content" id="cardAnswer"></div>
                        <div class="flashcard-source" id="cardSource"></div>
                    </div>
                </div>

                <div id="gradeButtons" class="grade-bar" style="display:none;">
                    <div class="grade-label">How well did you know it?</div>
                    <div class="grade-row">
                        <button class="grade-btn grade-0" onclick="submitGrade(0)">
                            <span class="grade-num">0</span>
                            <span class="grade-text">Blank</span>
                        </button>
                        <button class="grade-btn grade-1" onclick="submitGrade(1)">
                            <span class="grade-num">1</span>
                            <span class="grade-text">Wrong</span>
                        </button>
                        <button class="grade-btn grade-2" onclick="submitGrade(2)">
                            <span class="grade-num">2</span>
                            <span class="grade-text">Almost</span>
                        </button>
                        <button class="grade-btn grade-3" onclick="submitGrade(3)">
                            <span class="grade-num">3</span>
                            <span class="grade-text">Hard</span>
                        </button>
                        <button class="grade-btn grade-4" onclick="submitGrade(4)">
                            <span class="grade-num">4</span>
                            <span class="grade-text">Good</span>
                        </button>
                        <button class="grade-btn grade-5" onclick="submitGrade(5)">
                            <span class="grade-num">5</span>
                            <span class="grade-text">Easy</span>
                        </button>
                    </div>
                </div>
            </div>

            <div id="studyDone" class="empty-state" style="display:none;">
                <div style="font-size:1.5rem; margin-bottom:0.5rem;">&#10003;</div>
                Session complete! <span id="studyDoneCount"></span> questions reviewed.
                <div style="margin-top:1rem;">
                    <button class="btn btn-primary" onclick="loadStudy()">Review Again</button>
                </div>
            </div>

            <div id="upcomingSection" style="margin-top:2rem;">
                <h2 style="font-size:1rem; font-weight:600; color:#888; margin-bottom:1rem;">Upcoming Reviews</h2>
                <div id="upcomingList"></div>
            </div>
        </div>

        <!-- ======= QUESTION BANK VIEW ======= -->
        <div id="view-bank" class="view">
            <div class="stats-row" id="bankStats"></div>

            <div class="filter-bar" id="bankFilters">
                <button class="filter-btn active" onclick="setBankFilter('all', this)">All</button>
                <button class="filter-btn" onclick="setBankFilter('approved', this)">Approved</button>
                <button class="filter-btn" onclick="setBankFilter('pending', this)">Pending</button>
            </div>

            <div id="bankBatchBar" class="batch-bar" style="display:none;">
                <input type="checkbox" id="selectAllBank" onchange="toggleSelectAll('bank')">
                <span class="count" id="bankSelCount">0 selected</span>
                <button class="btn btn-approve btn-sm" onclick="batchAction('approve', 'bank')">Approve Selected</button>
                <button class="btn btn-delete btn-sm" onclick="batchAction('delete', 'bank')">Delete Selected</button>
            </div>
            <div id="bankList"></div>
        </div>
    </div>

    <script>
        const MAX_CHARS = 100000;
        let currentView = 'generate';
        let currentSubtab = 'text';
        let selectedFile = null;
        let bankFilter = 'all';

        // Track selections per context
        const selected = { pending: new Set(), bank: new Set() };

        // =====================
        // NAVIGATION
        // =====================
        function switchView(view) {
            currentView = view;
            document.querySelectorAll('.main-nav button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
            document.getElementById('view-' + view).classList.add('active');
            const btns = document.querySelectorAll('.main-nav button');
            const idx = ['generate', 'review', 'study', 'bank'].indexOf(view);
            btns[idx].classList.add('active');

            if (view === 'review') loadReview();
            if (view === 'study') loadStudy();
            if (view === 'bank') loadBank();
        }

        function switchSubtab(tab) {
            currentSubtab = tab;
            document.querySelectorAll('.subtab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.subtab-content').forEach(t => t.classList.remove('active'));
            document.getElementById('subtab-' + tab).classList.add('active');
            const tabs = document.querySelectorAll('.subtab');
            tabs[tab === 'text' ? 0 : 1].classList.add('active');
        }

        // =====================
        // GENERATE
        // =====================
        document.getElementById('source').addEventListener('input', function() {
            const len = this.value.length;
            const counter = document.getElementById('charCount');
            counter.textContent = len.toLocaleString() + ' / 100,000';
            counter.classList.toggle('warn', len > MAX_CHARS * 0.9);
        });

        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files[0]) selectPdf(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files[0]) selectPdf(fileInput.files[0]); });

        function selectPdf(file) {
            if (file.type !== 'application/pdf') {
                document.getElementById('genStatus').textContent = 'Error: Only PDF files.';
                return;
            }
            if (file.size > 20 * 1024 * 1024) {
                document.getElementById('genStatus').textContent = 'Error: PDF exceeds 20MB.';
                return;
            }
            selectedFile = file;
            document.getElementById('fileName').textContent = file.name + ' (' + (file.size / 1024).toFixed(0) + ' KB)';
            // Auto-fill title from filename
            const titleInput = document.getElementById('sourceTitle');
            if (!titleInput.value.trim()) {
                titleInput.value = file.name.replace(/\\.pdf$/i, '');
            }
            document.getElementById('genStatus').textContent = '';
        }

        async function generate() {
            const btn = document.getElementById('genBtn');
            const status = document.getElementById('genStatus');
            const title = document.getElementById('sourceTitle').value.trim();

            btn.disabled = true;
            btn.textContent = 'Generating...';

            try {
                let res;
                if (currentSubtab === 'text') {
                    const text = document.getElementById('source').value.trim();
                    if (!text) { status.textContent = 'Paste some text first.'; return; }
                    status.textContent = 'Sending to Groq API...';
                    res = await fetch('/generate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text, title: title || 'Untitled' })
                    });
                } else {
                    if (!selectedFile) { status.textContent = 'Select a PDF first.'; return; }
                    status.textContent = 'Uploading PDF and extracting text...';
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    res = await fetch('/upload-pdf', { method: 'POST', body: formData });
                }

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.error || 'Request failed');
                }

                const data = await res.json();
                const count = data.questions ? data.questions.length : 0;

                if (count === 0) {
                    status.textContent = 'No questions generated. Try different material.';
                    return;
                }

                let info = count + ' questions generated from ' + data.chunk_count + ' chunk(s).';
                if (data.extracted_chars) info += ' ' + data.extracted_chars.toLocaleString() + ' chars extracted.';
                info += ' Go to Review to vet them.';
                status.textContent = info;

                // Clear inputs
                document.getElementById('source').value = '';
                document.getElementById('charCount').textContent = '0 / 100,000';
                document.getElementById('sourceTitle').value = '';
                selectedFile = null;
                document.getElementById('fileName').textContent = '';
                fileInput.value = '';

                updateBadges();

            } catch (e) {
                status.textContent = 'Error: ' + e.message;
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Questions';
            }
        }

        // =====================
        // REVIEW (pending only)
        // =====================
        async function loadReview() {
            const list = document.getElementById('reviewList');
            list.innerHTML = '';
            selected.pending.clear();
            updateSelectionCount('pending');
            document.getElementById('selectAllPending').checked = false;

            try {
                const questions = await fetch('/api/questions?status=pending').then(r => r.json());
                document.getElementById('pendingBadge').textContent = questions.length;

                if (questions.length === 0) {
                    list.innerHTML = '<div class="empty-state">No pending questions. Generate some from the Generate tab.</div>';
                    document.getElementById('reviewBatchBar').style.display = 'none';
                    return;
                }

                document.getElementById('reviewBatchBar').style.display = 'flex';
                const frag = document.createDocumentFragment();
                questions.forEach(q => frag.appendChild(makeCard(q, 'pending')));
                list.appendChild(frag); // single DOM insertion = single reflow
            } catch (e) {
                list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
            }
        }

        // =====================
        // QUESTION BANK (all)
        // =====================
        async function loadBank() {
            const list = document.getElementById('bankList');
            const statsEl = document.getElementById('bankStats');
            list.innerHTML = '';
            selected.bank.clear();
            updateSelectionCount('bank');
            document.getElementById('selectAllBank').checked = false;

            try {
                const statusParam = bankFilter === 'all' ? '' : '?status=' + bankFilter;
                // Parallel: question list + cheap COUNT stats (no second full-table download)
                const [questions, counts] = await Promise.all([
                    fetch('/api/questions' + statusParam).then(r => r.json()),
                    fetch('/api/counts').then(r => r.json()),
                ]);
                const approved = counts.approved;
                const pending = counts.pending;
                const total = counts.total;

                document.getElementById('approvedBadge').textContent = approved;
                document.getElementById('pendingBadge').textContent = pending;

                statsEl.innerHTML = '';
                [{num: total, label: 'Total'}, {num: approved, label: 'Approved'}, {num: pending, label: 'Pending'}].forEach(s => {
                    const box = document.createElement('div');
                    box.className = 'stat-box';
                    const numDiv = document.createElement('div');
                    numDiv.className = 'num';
                    numDiv.textContent = s.num;
                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'label';
                    labelDiv.textContent = s.label;
                    box.appendChild(numDiv);
                    box.appendChild(labelDiv);
                    statsEl.appendChild(box);
                });

                if (questions.length === 0) {
                    list.innerHTML = '<div class="empty-state">No questions found.</div>';
                    document.getElementById('bankBatchBar').style.display = 'none';
                    return;
                }

                document.getElementById('bankBatchBar').style.display = 'flex';
                const frag = document.createDocumentFragment();
                questions.forEach(q => frag.appendChild(makeCard(q, 'bank')));
                list.appendChild(frag);
            } catch (e) {
                list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
            }
        }

        function setBankFilter(f, btn) {
            bankFilter = f;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active'); // explicit element instead of deprecated global `event`
            loadBank();
        }

        // =====================
        // CARD RENDERING
        // =====================
        function makeCard(q, context) {
            const card = document.createElement('div');
            card.className = 'question-card';
            card.dataset.id = q.id;
            card.id = 'card-' + q.id;

            const header = document.createElement('div');
            header.className = 'card-header';

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.onchange = () => toggleSelect(q.id, context);
            if (selected[context].has(q.id)) cb.checked = true;

            const body = document.createElement('div');
            body.className = 'card-body';

            const qDiv = document.createElement('div');
            qDiv.className = 'q';
            qDiv.textContent = q.question;

            const aDiv = document.createElement('div');
            aDiv.className = 'a';
            aDiv.textContent = q.answer;

            const meta = document.createElement('div');
            meta.className = 'meta';

            const typeTag = document.createElement('span');
            typeTag.className = 'tag';
            typeTag.textContent = q.type || 'unknown';
            meta.appendChild(typeTag);

            if (q.status) {
                const statusTag = document.createElement('span');
                statusTag.className = 'tag ' + q.status;
                statusTag.textContent = q.status;
                meta.appendChild(statusTag);
            }

            if (q.source_title) {
                const srcSpan = document.createElement('span');
                srcSpan.className = 'source-label';
                srcSpan.textContent = q.source_title;
                meta.appendChild(srcSpan);
            }

            const actions = document.createElement('div');
            actions.className = 'actions';

            if (q.status === 'pending') {
                const approveBtn = document.createElement('button');
                approveBtn.className = 'btn btn-approve btn-sm';
                approveBtn.textContent = 'Approve';
                approveBtn.onclick = () => updateStatus(q.id, 'approved', context);
                actions.appendChild(approveBtn);
            }

            const editBtn = document.createElement('button');
            editBtn.className = 'btn btn-edit btn-sm';
            editBtn.textContent = 'Edit';
            editBtn.onclick = () => startEdit(q.id, q.question, q.answer);
            actions.appendChild(editBtn);

            const delBtn = document.createElement('button');
            delBtn.className = 'btn btn-delete btn-sm';
            delBtn.textContent = 'Delete';
            delBtn.onclick = () => deleteQuestion(q.id, context);
            actions.appendChild(delBtn);

            body.appendChild(qDiv);
            body.appendChild(aDiv);
            body.appendChild(meta);
            body.appendChild(actions);
            header.appendChild(cb);
            header.appendChild(body);
            card.appendChild(header);

            return card;
        }

        // =====================
        // CRUD ACTIONS
        // =====================
        function removeCardInPlace(id, context) {
            // Remove single card instead of re-fetching + re-rendering the whole list
            const card = document.getElementById('card-' + id);
            if (card) card.remove();
            selected[context].delete(id);
            updateSelectionCount(context);
        }

        async function updateStatus(id, status, context) {
            try {
                const res = await fetch('/api/questions/' + id, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status })
                });
                if (!res.ok) throw new Error('Failed');
                if (context === 'pending') removeCardInPlace(id, context); // approved -> leaves pending list
                else loadBank();
                updateBadges();
            } catch (e) { console.error(e); }
        }

        async function deleteQuestion(id, context) {
            try {
                const res = await fetch('/api/questions/' + id, { method: 'DELETE' });
                if (!res.ok) throw new Error('Failed');
                removeCardInPlace(id, context);
                updateBadges();
            } catch (e) { console.error(e); }
        }

        function startEdit(id, question, answer) {
            const card = document.getElementById('card-' + id);
            const body = card.querySelector('.card-body');
            body.innerHTML = '';

            const qInput = document.createElement('input');
            qInput.type = 'text';
            qInput.className = 'edit-field';
            qInput.value = question;
            qInput.placeholder = 'Question';

            const aInput = document.createElement('textarea');
            aInput.className = 'edit-field';
            aInput.value = answer;
            aInput.placeholder = 'Answer';

            const btnRow = document.createElement('div');
            btnRow.className = 'actions';

            const saveBtn = document.createElement('button');
            saveBtn.className = 'btn btn-primary btn-sm';
            saveBtn.textContent = 'Save';
            saveBtn.onclick = () => saveEdit(id, qInput.value, aInput.value);

            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-ghost btn-sm';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.onclick = () => {
                if (currentView === 'review') loadReview();
                else loadBank();
            };

            btnRow.appendChild(saveBtn);
            btnRow.appendChild(cancelBtn);
            body.appendChild(qInput);
            body.appendChild(aInput);
            body.appendChild(btnRow);
            qInput.focus();
        }

        async function saveEdit(id, question, answer) {
            question = question.trim();
            answer = answer.trim();
            if (!question || !answer) return;

            try {
                const res = await fetch('/api/questions/' + id, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, answer })
                });
                if (!res.ok) throw new Error('Failed');
                if (currentView === 'review') loadReview();
                else loadBank();
            } catch (e) { console.error(e); }
        }

        // =====================
        // SELECTION & BATCH
        // =====================
        function toggleSelect(id, context) {
            if (selected[context].has(id)) {
                selected[context].delete(id);
            } else {
                selected[context].add(id);
            }
            updateSelectionCount(context);
            const card = document.getElementById('card-' + id);
            if (card) card.classList.toggle('selected', selected[context].has(id));
        }

        function toggleSelectAll(context) {
            const checkboxId = context === 'pending' ? 'selectAllPending' : 'selectAllBank';
            const listId = context === 'pending' ? 'reviewList' : 'bankList';
            const checked = document.getElementById(checkboxId).checked;

            const cards = document.getElementById(listId).querySelectorAll('.question-card');
            cards.forEach(card => {
                const id = parseInt(card.dataset.id);
                const cb = card.querySelector('input[type="checkbox"]');
                if (checked) {
                    selected[context].add(id);
                    if (cb) cb.checked = true;
                    card.classList.add('selected');
                } else {
                    selected[context].delete(id);
                    if (cb) cb.checked = false;
                    card.classList.remove('selected');
                }
            });
            updateSelectionCount(context);
        }

        function updateSelectionCount(context) {
            const countEl = context === 'pending' ? 'reviewSelCount' : 'bankSelCount';
            document.getElementById(countEl).textContent = selected[context].size + ' selected';
        }

        async function batchAction(action, context) {
            const ids = Array.from(selected[context]);
            if (ids.length === 0) return;

            try {
                const res = await fetch('/api/questions/batch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ids, action })
                });
                if (!res.ok) throw new Error('Failed');
                selected[context].clear();
                if (context === 'pending') loadReview();
                else loadBank();
                updateBadges();
            } catch (e) { console.error(e); }
        }

        // =====================
        // STUDY SESSION
        // =====================
        let studyQueue = [];
        let studyIndex = 0;
        let studyReviewed = 0;
        let cardFlipped = false;
        let isPaused = false;

        async function togglePause() {
            try {
                if (isPaused) {
                    await fetch('/api/resume', { method: 'POST' });
                } else {
                    await fetch('/api/pause', { method: 'POST' });
                }
                loadStudy();
                updateBadges();
            } catch (e) {
                console.error('Pause toggle failed:', e);
            }
        }

        function updatePauseUI(stats) {
            const bar = document.getElementById('pauseBar');
            const btn = document.getElementById('pauseBtn');
            const status = document.getElementById('pauseStatus');

            // Show pause bar if there are any approved questions
            if (stats.total_in_rotation > 0) {
                bar.style.display = 'flex';
                bar.style.alignItems = 'center';
                bar.style.gap = '0.8rem';
            } else {
                bar.style.display = 'none';
                return;
            }

            isPaused = stats.paused > 0;
            if (isPaused) {
                btn.textContent = 'Resume Schedule';
                btn.classList.add('paused');
                status.textContent = stats.paused + ' questions paused';
            } else {
                btn.textContent = 'Pause Schedule';
                btn.classList.remove('paused');
                status.textContent = '';
            }
        }

        async function loadStudy() {
            const emptyEl = document.getElementById('studyEmpty');
            const sessionEl = document.getElementById('studySession');
            const doneEl = document.getElementById('studyDone');
            const statsEl = document.getElementById('studyStats');

            emptyEl.style.display = 'none';
            sessionEl.style.display = 'none';
            doneEl.style.display = 'none';

            try {
                // Stats + due queue in parallel; upcoming list loads independently
                const [stats, due] = await Promise.all([
                    fetch('/api/stats').then(r => r.json()),
                    fetch('/api/due').then(r => r.json()),
                ]);
                statsEl.innerHTML = '';
                [
                    { num: stats.due_now, label: 'Due Now' },
                    { num: stats.total_in_rotation, label: 'In Rotation' },
                    { num: stats.reviewed_at_least_once, label: 'Reviewed' },
                    { num: stats.upcoming_7_days, label: 'Next 7 Days' },
                ].forEach(s => {
                    const box = document.createElement('div');
                    box.className = 'stat-box';
                    const numDiv = document.createElement('div');
                    numDiv.className = 'num';
                    numDiv.textContent = s.num;
                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'label';
                    labelDiv.textContent = s.label;
                    box.appendChild(numDiv);
                    box.appendChild(labelDiv);
                    statsEl.appendChild(box);
                });

                updatePauseUI(stats);

                // True count from stats — due.length is capped at 20 by the API
                document.getElementById('dueBadge').textContent = stats.due_now;

                if (due.length === 0) {
                    emptyEl.style.display = 'block';
                    emptyEl.textContent = 'No questions due right now. Check back later or approve more questions.';
                    loadUpcoming();
                    return;
                }

                studyQueue = due;
                studyIndex = 0;
                studyReviewed = 0;
                sessionEl.style.display = 'block';
                showCard();
                loadUpcoming();

            } catch (e) {
                emptyEl.style.display = 'block';
                emptyEl.textContent = 'Failed to load study session.';
            }
        }

        function showCard() {
            if (studyIndex >= studyQueue.length) {
                document.getElementById('studySession').style.display = 'none';
                document.getElementById('studyDone').style.display = 'block';
                document.getElementById('studyDoneCount').textContent = studyReviewed;
                updateBadges();
                loadUpcoming();
                return;
            }

            const q = studyQueue[studyIndex];
            cardFlipped = false;

            document.getElementById('studyProgress').textContent =
                (studyIndex + 1) + ' / ' + studyQueue.length;

            document.getElementById('cardQuestion').textContent = q.question;
            document.getElementById('cardAnswer').textContent = q.answer;
            document.getElementById('cardSource').textContent = q.source_title || '';

            document.getElementById('cardFront').style.display = 'block';
            document.getElementById('cardBack').style.display = 'none';
            document.getElementById('gradeButtons').style.display = 'none';
        }

        function flipCard() {
            if (cardFlipped) return;
            cardFlipped = true;
            document.getElementById('cardFront').style.display = 'none';
            document.getElementById('cardBack').style.display = 'block';
            document.getElementById('gradeButtons').style.display = 'block';
        }

        function submitGrade(quality) {
            const q = studyQueue[studyIndex];
            // Optimistic: advance the UI immediately, send the grade in the background
            fetch('/api/review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question_id: q.id, quality })
            }).catch(e => console.error('Failed to submit grade:', e));

            studyReviewed++;
            studyIndex++;
            showCard();
        }

        // =====================
        // UPCOMING REVIEWS
        // =====================
        async function loadUpcoming() {
            const listEl = document.getElementById('upcomingList');
            const sectionEl = document.getElementById('upcomingSection');
            listEl.innerHTML = '';

            try {
                const upcoming = await fetch('/api/upcoming').then(r => r.json());

                if (upcoming.length === 0) {
                    sectionEl.style.display = 'none';
                    return;
                }

                sectionEl.style.display = 'block';

                // Group by day
                const grouped = {};
                const now = new Date();
                upcoming.forEach(q => {
                    const d = new Date(q.next_review);
                    const diffDays = Math.ceil((d - now) / (1000 * 60 * 60 * 24));
                    let label;
                    if (diffDays <= 0) label = 'Today';
                    else if (diffDays === 1) label = 'Tomorrow';
                    else {
                        const opts = { weekday: 'short', month: 'short', day: 'numeric' };
                        label = d.toLocaleDateString(undefined, opts);
                    }
                    if (!grouped[label]) grouped[label] = [];
                    grouped[label].push(q);
                });

                const upFrag = document.createDocumentFragment();
                for (const [day, questions] of Object.entries(grouped)) {
                    const dayDiv = document.createElement('div');
                    dayDiv.className = 'upcoming-day';

                    const header = document.createElement('div');
                    header.className = 'upcoming-day-header';
                    header.textContent = day + ' (' + questions.length + ')';
                    dayDiv.appendChild(header);

                    questions.forEach(q => {
                        const item = document.createElement('div');
                        item.className = 'upcoming-item';

                        const qText = document.createElement('div');
                        qText.className = 'upcoming-q';
                        qText.textContent = q.question;

                        const meta = document.createElement('div');
                        meta.className = 'upcoming-meta';

                        if (q.source_title) {
                            const src = document.createElement('span');
                            src.className = 'upcoming-interval';
                            src.textContent = q.source_title;
                            meta.appendChild(src);
                        }

                        const ivTag = document.createElement('span');
                        ivTag.className = 'upcoming-interval';
                        ivTag.textContent = q.interval + 'd interval';
                        meta.appendChild(ivTag);

                        item.appendChild(qText);
                        item.appendChild(meta);
                        dayDiv.appendChild(item);
                    });

                    upFrag.appendChild(dayDiv);
                }
                listEl.appendChild(upFrag);
            } catch (e) {
                sectionEl.style.display = 'none';
            }
        }

        // =====================
        // BADGE COUNTS
        // =====================
        async function updateBadges() {
            // One tiny COUNT query instead of downloading the entire question table
            try {
                const c = await fetch('/api/counts').then(r => r.json());
                document.getElementById('pendingBadge').textContent = c.pending;
                document.getElementById('approvedBadge').textContent = c.approved;
                document.getElementById('dueBadge').textContent = c.due;
            } catch (e) {}
        }

        // Init badges on load
        updateBadges();

        // Keyboard shortcuts for study
        document.addEventListener('keydown', function(e) {
            if (currentView !== 'study') return;
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.code === 'Space' && !cardFlipped) {
                e.preventDefault();
                flipCard();
            } else if (cardFlipped && e.key >= '0' && e.key <= '5') {
                e.preventDefault();
                submitGrade(parseInt(e.key));
            }
        });
    </script>
</body>
</html>
"""
