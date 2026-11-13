import asyncio
import json
import logging
import os
import sqlite3
import sys
import textwrap
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
    telegramInstalled = True
except ImportError:
    telegramInstalled = False

load_dotenv()

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


botToken = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
chatId = os.getenv("TELEGRAM_CHAT_ID", "").strip()

telegramEnabled = bool(
    telegramInstalled
    and botToken
    and botToken != "your_token_here"
    and chatId
)

threeHoursInSeconds = 3 * 60 * 60
nudgeIntervalSeconds = threeHoursInSeconds
studyHoursStart = 8
studyHoursEnd = 23

if telegramInstalled and not telegramEnabled:
    logger.warning("Telegram bot disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")


groqApiKey = os.getenv("GROQ_API_KEY", "").strip()
if not groqApiKey or groqApiKey == "your_key_here":
    print("ERROR: Set a valid GROQ_API_KEY in .env")
    sys.exit(1)


@asynccontextmanager
async def lifespan(fastApiApp: FastAPI):
    await startTelegramBot()
    yield
    await stopTelegramBot()


app = FastAPI(lifespan=lifespan)
groqClient = Groq(api_key=groqApiKey)


databasePath = "spaced_rep.db"


def getDatabase():
    connection = sqlite3.connect(databasePath, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def utcNow():
    return datetime.now(timezone.utc)


def utcNowText():
    return utcNow().isoformat()


def getColumnNames(connection, tableName):
    columnRows = connection.execute(f"PRAGMA table_info({tableName})").fetchall()
    columnNames = set()
    for columnRow in columnRows:
        columnNames.add(columnRow[1])
    return columnNames


oldColumnNames = {
    "sources": {
        "char_count": "charCount",
        "created_at": "createdAt",
        "topic_name": "topicName",
    },
    "questions": {
        "source_id": "sourceId",
        "created_at": "createdAt",
        "easiness_factor": "easinessFactor",
        "next_review": "nextReview",
    },
}

oldIndexNames = [
    "idx_questions_status",
    "idx_questions_source",
    "idx_questions_next_review",
    "idx_questions_due",
]


def renameOldColumns(connection):
    for tableName, renames in oldColumnNames.items():
        existingColumns = getColumnNames(connection, tableName)
        for oldName, newName in renames.items():
            if oldName in existingColumns:
                connection.execute(f"ALTER TABLE {tableName} RENAME COLUMN {oldName} TO {newName}")

    for indexName in oldIndexNames:
        connection.execute(f"DROP INDEX IF EXISTS {indexName}")


def addMissingColumns(connection):
    questionColumns = getColumnNames(connection, "questions")
    reviewColumns = {
        "easinessFactor": "REAL NOT NULL DEFAULT 2.5",
        "interval": "INTEGER NOT NULL DEFAULT 0",
        "repetitions": "INTEGER NOT NULL DEFAULT 0",
        "nextReview": "TEXT",
        "paused": "INTEGER NOT NULL DEFAULT 0",
    }
    for columnName, columnType in reviewColumns.items():
        if columnName not in questionColumns:
            connection.execute(f"ALTER TABLE questions ADD COLUMN {columnName} {columnType}")

    sourceColumns = getColumnNames(connection, "sources")
    if "topicName" not in sourceColumns:
        connection.execute("ALTER TABLE sources ADD COLUMN topicName TEXT")


def createTables():
    connection = getDatabase()
    connection.execute("PRAGMA journal_mode=WAL")

    connection.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'text',
            charCount INTEGER NOT NULL DEFAULT 0,
            createdAt TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sourceId INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'unknown',
            status TEXT NOT NULL DEFAULT 'pending',
            createdAt TEXT NOT NULL,
            FOREIGN KEY (sourceId) REFERENCES sources(id) ON DELETE SET NULL
        );
    """)

    renameOldColumns(connection)
    addMissingColumns(connection)

    connection.execute("CREATE INDEX IF NOT EXISTS idxQuestionsStatus ON questions(status)")
    connection.execute("CREATE INDEX IF NOT EXISTS idxQuestionsSource ON questions(sourceId)")
    connection.execute("CREATE INDEX IF NOT EXISTS idxQuestionsNextReview ON questions(nextReview)")
    connection.execute("CREATE INDEX IF NOT EXISTS idxQuestionsDue ON questions(status, paused, nextReview)")

    connection.commit()
    connection.close()


createTables()


questionGeneratorInstructions = textwrap.dedent("""\
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

topicNameInstructions = (
    "Given source material, respond with ONLY a short topic name (2-5 words) that describes the subject. "
    "Examples: 'Digital Logic', 'Cell Biology', 'French Revolution', 'Linear Algebra'. "
    "No quotes, no punctuation, no explanation."
)

chunkSize = 8000
chunkOverlap = 500
groqModel = "llama-3.3-70b-versatile"
maxParallelGroqCalls = 4

maxInputCharacters = 100_000
maxChunks = 15
maxPdfBytes = 20 * 1024 * 1024
maxPdfPages = 200

allowedQuestionTypes = ["factual", "conceptual", "application"]
allowedStatuses = ["pending", "approved"]


def splitIntoChunks(text: str) -> list[str]:
    chunks = []
    start = 0
    textLength = len(text)

    while start < textLength and len(chunks) < maxChunks:
        end = min(start + chunkSize, textLength)
        chunks.append(text[start:end])
        if end >= textLength:
            break
        start = end - chunkOverlap

    return chunks


def extractPdfText(pdfBytes: bytes) -> str:
    document = fitz.open(stream=pdfBytes, filetype="pdf")

    if document.page_count > maxPdfPages:
        document.close()
        raise ValueError(f"PDF has {document.page_count} pages. Max is {maxPdfPages}.")

    characterLimit = maxInputCharacters * 2
    pageTexts = []
    totalCharacters = 0

    for page in document:
        pageText = page.get_text()
        charactersBefore = totalCharacters
        totalCharacters += len(pageText)

        if totalCharacters > characterLimit:
            charactersLeft = characterLimit - charactersBefore
            pageTexts.append(pageText[:charactersLeft])
            break

        pageTexts.append(pageText)

    document.close()
    return "\n".join(pageTexts)


def cleanQuestions(parsedReply: dict) -> list[dict]:
    questions = parsedReply.get("questions")
    if not isinstance(questions, list):
        return []

    cleanedQuestions = []
    for item in questions:
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        answer = item.get("answer")
        questionType = item.get("type", "unknown")

        if not isinstance(question, str) or not isinstance(answer, str):
            continue
        if not question.strip() or not answer.strip():
            continue
        if questionType not in allowedQuestionTypes:
            questionType = "unknown"

        cleanedQuestions.append({
            "question": question.strip()[:1000],
            "answer": answer.strip()[:2000],
            "type": questionType,
        })

    return cleanedQuestions


def isRateLimitError(error: Exception) -> bool:
    errorText = str(error)
    return "429" in errorText or "rate" in errorText.lower()


def askGroqForQuestions(textChunk: str, retries: int = 3) -> list[dict]:
    for attempt in range(retries):
        isLastAttempt = attempt == retries - 1

        try:
            response = groqClient.chat.completions.create(
                model=groqModel,
                messages=[
                    {"role": "system", "content": questionGeneratorInstructions},
                    {"role": "user", "content": f"Generate study questions from this material:\n\n{textChunk}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            replyText = response.choices[0].message.content
            logger.debug("Groq raw response: %s", replyText[:500])
            parsedReply = json.loads(replyText)
            return cleanQuestions(parsedReply)

        except json.JSONDecodeError:
            logger.warning("Groq returned invalid JSON on attempt %d", attempt + 1)
            if isLastAttempt:
                return []

        except Exception as error:
            if isRateLimitError(error):
                waitSeconds = (2 ** attempt) * 2
                logger.info("Rate limited, waiting %ds...", waitSeconds)
                time.sleep(waitSeconds)
            else:
                errorType = type(error).__name__
                shortMessage = str(error)[:200]
                logger.warning("Groq API error (attempt %d): [%s] %s", attempt + 1, errorType, shortMessage)
                if isLastAttempt:
                    return []

    return []


def askGroqForTopicName(textSample: str) -> str:
    try:
        response = groqClient.chat.completions.create(
            model=groqModel,
            messages=[
                {"role": "system", "content": topicNameInstructions},
                {"role": "user", "content": textSample[:3000]},
            ],
            temperature=0.1,
        )
        topicName = response.choices[0].message.content.strip().strip('"\'.')
        if topicName:
            return topicName[:100]
        return "Unknown Topic"
    except Exception as error:
        logger.warning(f"Failed to generate topic name: {error}")
        return "Unknown Topic"


def saveSourceAndQuestions(title, sourceType, characterCount, topicName, questions):
    now = utcNowText()
    connection = getDatabase()
    try:
        cursor = connection.execute(
            "INSERT INTO sources (title, type, charCount, createdAt, topicName) VALUES (?, ?, ?, ?, ?)",
            (title, sourceType, characterCount, now, topicName),
        )
        sourceId = cursor.lastrowid

        questionRows = []
        for question in questions:
            questionRows.append((sourceId, question["question"], question["answer"], question["type"], now))

        connection.executemany(
            "INSERT INTO questions (sourceId, question, answer, type, status, createdAt) VALUES (?, ?, ?, ?, 'pending', ?)",
            questionRows,
        )
        connection.commit()
    finally:
        connection.close()

    return sourceId


async def generateQuestionsFromText(text: str, sourceTitle: str, sourceType: str = "text") -> dict:
    chunks = splitIntoChunks(text)
    limiter = asyncio.Semaphore(maxParallelGroqCalls)

    async def processChunk(chunk: str) -> list[dict]:
        async with limiter:
            return await asyncio.to_thread(askGroqForQuestions, chunk)

    topicNameTask = asyncio.create_task(asyncio.to_thread(askGroqForTopicName, text))

    chunkTasks = []
    for chunk in chunks:
        chunkTasks.append(processChunk(chunk))
    questionsPerChunk = await asyncio.gather(*chunkTasks)

    topicName = await topicNameTask

    allQuestions = []
    for chunkQuestions in questionsPerChunk:
        allQuestions.extend(chunkQuestions)

    sourceId = saveSourceAndQuestions(sourceTitle, sourceType, len(text), topicName, allQuestions)

    return {
        "questions": allQuestions,
        "chunkCount": len(chunks),
        "sourceId": sourceId,
        "topicName": topicName,
    }


maxIntervalDays = 365


def calculateNextReview(quality: int, easinessFactor: float, interval: int, repetitions: int) -> tuple[float, int, int]:
    quality = max(0, min(5, quality))

    mistakes = 5 - quality
    newEasinessFactor = easinessFactor + (0.1 - mistakes * (0.08 + mistakes * 0.02))
    newEasinessFactor = max(1.3, newEasinessFactor)

    answeredCorrectly = quality >= 3
    if not answeredCorrectly:
        return newEasinessFactor, 1, 0

    newRepetitions = repetitions + 1
    if newRepetitions == 1:
        newInterval = 1
    elif newRepetitions == 2:
        newInterval = 3
    else:
        newInterval = round(interval * newEasinessFactor)

    newInterval = min(newInterval, maxIntervalDays)

    return newEasinessFactor, newInterval, newRepetitions


def gradeQuestion(connection, questionId: int, quality: int):
    row = connection.execute(
        "SELECT easinessFactor, interval, repetitions FROM questions WHERE id = ?",
        (questionId,),
    ).fetchone()
    if not row:
        return None

    newEasinessFactor, newInterval, newRepetitions = calculateNextReview(
        quality, row["easinessFactor"], row["interval"], row["repetitions"]
    )
    nextReview = (utcNow() + timedelta(days=newInterval)).isoformat()

    connection.execute(
        "UPDATE questions SET easinessFactor = ?, interval = ?, repetitions = ?, nextReview = ? WHERE id = ?",
        (newEasinessFactor, newInterval, newRepetitions, nextReview, questionId),
    )
    connection.commit()

    return {
        "easinessFactor": newEasinessFactor,
        "interval": newInterval,
        "repetitions": newRepetitions,
        "nextReview": nextReview,
    }


maxJsonBodyBytes = 512 * 1024


def errorResponse(message, statusCode):
    return JSONResponse({"error": message}, status_code=statusCode)


async def readJsonBody(request: Request):
    bodyBytes = await request.body()
    if len(bodyBytes) > maxJsonBodyBytes:
        return None, errorResponse("Request body too large", 413)

    try:
        body = json.loads(bodyBytes)
    except json.JSONDecodeError:
        return None, errorResponse("Invalid JSON", 400)

    if not isinstance(body, dict):
        return None, errorResponse("JSON body must be an object", 400)

    return body, None


@app.post("/generate")
async def generateFromPastedText(request: Request):
    body, error = await readJsonBody(request)
    if error:
        return error

    text = body.get("text", "")
    title = body.get("title", "")

    if not isinstance(title, str):
        title = ""
    title = title.strip() or "Untitled"

    if not isinstance(text, str):
        return errorResponse("text must be a string", 400)

    text = text.strip()
    if not text:
        return errorResponse("No text provided", 400)

    if len(text) > maxInputCharacters:
        return errorResponse(f"Input too long ({len(text)} chars). Max is {maxInputCharacters}.", 413)

    return await generateQuestionsFromText(text, title[:200], "text")


async def readUploadWithLimit(file: UploadFile):
    fileBytes = bytearray()
    while True:
        piece = await file.read(1024 * 1024)
        if not piece:
            break
        fileBytes.extend(piece)
        if len(fileBytes) > maxPdfBytes:
            return None
    return bytes(fileBytes)


def cleanFileName(fileName):
    title = fileName or "Uploaded PDF"
    title = title.rsplit("/", 1)[-1]
    title = title.rsplit("\\", 1)[-1]

    printableCharacters = []
    for character in title:
        if character.isprintable():
            printableCharacters.append(character)
    title = "".join(printableCharacters)

    return title.strip() or "Uploaded PDF"


@app.post("/upload-pdf")
async def generateFromPdf(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return errorResponse("File must be a PDF", 400)

    pdfBytes = await readUploadWithLimit(file)
    if pdfBytes is None:
        return errorResponse("PDF too large. Max is 20MB.", 413)

    try:
        text = await asyncio.to_thread(extractPdfText, pdfBytes)
    except ValueError as error:
        return errorResponse(str(error), 413)
    except Exception:
        return errorResponse("Failed to read PDF. File may be corrupted or encrypted.", 400)

    text = text.strip()
    if not text:
        return errorResponse("No text found in PDF. It may be scanned images without OCR.", 400)

    text = text[:maxInputCharacters]

    title = cleanFileName(file.filename)
    result = await generateQuestionsFromText(text, title[:200], "pdf")
    result["extractedChars"] = len(text)
    return result


questionWithSourceQuery = """
    SELECT questions.*, sources.title AS sourceTitle
    FROM questions
    LEFT JOIN sources ON questions.sourceId = sources.id
"""

dueCondition = "status = 'approved' AND paused = 0 AND (nextReview IS NULL OR nextReview <= ?)"


def rowsToList(rows):
    return [dict(row) for row in rows]


@app.get("/api/questions")
def listQuestions(status: str = None, sourceId: int = None):
    connection = getDatabase()
    try:
        query = questionWithSourceQuery + " WHERE 1=1"
        parameters = []

        if status:
            if status not in allowedStatuses:
                return errorResponse("Invalid status filter", 400)
            query += " AND questions.status = ?"
            parameters.append(status)

        if sourceId is not None:
            query += " AND questions.sourceId = ?"
            parameters.append(sourceId)

        query += " ORDER BY questions.createdAt DESC"
        return rowsToList(connection.execute(query, parameters).fetchall())
    finally:
        connection.close()


def questionExists(connection, questionId):
    row = connection.execute("SELECT id FROM questions WHERE id = ?", (questionId,)).fetchone()
    return row is not None


def readValidEditValue(fieldName, rawValue):
    value = str(rawValue).strip()

    if fieldName == "question":
        return value[:1000]
    if fieldName == "answer":
        return value[:2000]
    if fieldName == "status" and value not in allowedStatuses:
        return None
    if fieldName == "type" and value not in allowedQuestionTypes + ["unknown"]:
        return None
    return value


@app.put("/api/questions/{questionId}")
async def updateQuestion(questionId: int, request: Request):
    body, error = await readJsonBody(request)
    if error:
        return error

    connection = getDatabase()
    try:
        if not questionExists(connection, questionId):
            return errorResponse("Question not found", 404)

        setParts = []
        parameters = []
        for fieldName in ["question", "answer", "type", "status"]:
            if fieldName not in body:
                continue
            value = readValidEditValue(fieldName, body[fieldName])
            if value is None:
                continue
            setParts.append(f"{fieldName} = ?")
            parameters.append(value)

        if not setParts:
            return errorResponse("No valid fields to update", 400)

        if body.get("status") == "approved":
            setParts.append("nextReview = COALESCE(nextReview, ?)")
            parameters.append(utcNowText())

        parameters.append(questionId)
        connection.execute(f"UPDATE questions SET {', '.join(setParts)} WHERE id = ?", parameters)
        connection.commit()

        row = connection.execute(questionWithSourceQuery + " WHERE questions.id = ?", (questionId,)).fetchone()
        return dict(row)
    finally:
        connection.close()


@app.delete("/api/questions/{questionId}")
def deleteQuestion(questionId: int):
    connection = getDatabase()
    try:
        if not questionExists(connection, questionId):
            return errorResponse("Question not found", 404)

        connection.execute("DELETE FROM questions WHERE id = ?", (questionId,))
        connection.commit()
        return {"deleted": questionId}
    finally:
        connection.close()


@app.post("/api/questions/batch")
async def batchUpdate(request: Request):
    body, error = await readJsonBody(request)
    if error:
        return error

    ids = body.get("ids", [])
    action = body.get("action", "")

    if not isinstance(ids, list) or not ids:
        return errorResponse("ids must be a non-empty list", 400)
    if len(ids) > 500:
        return errorResponse("Too many ids. Max is 500 per batch.", 400)
    if action not in ("approve", "delete"):
        return errorResponse("action must be 'approve' or 'delete'", 400)

    try:
        ids = [int(questionId) for questionId in ids]
    except (ValueError, TypeError):
        return errorResponse("ids must be integers", 400)

    placeholders = ",".join(["?"] * len(ids))

    connection = getDatabase()
    try:
        if action == "approve":
            connection.execute(
                f"UPDATE questions SET status = 'approved', nextReview = COALESCE(nextReview, ?) WHERE id IN ({placeholders})",
                [utcNowText()] + ids,
            )
        else:
            connection.execute(f"DELETE FROM questions WHERE id IN ({placeholders})", ids)

        connection.commit()
        return {"action": action, "count": len(ids)}
    finally:
        connection.close()


@app.get("/api/sources")
def listSources():
    connection = getDatabase()
    try:
        rows = connection.execute("""
            SELECT sources.*,
                   COUNT(questions.id) AS totalQuestions,
                   SUM(CASE WHEN questions.status = 'approved' THEN 1 ELSE 0 END) AS approvedCount,
                   SUM(CASE WHEN questions.status = 'pending' THEN 1 ELSE 0 END) AS pendingCount
            FROM sources
            LEFT JOIN questions ON questions.sourceId = sources.id
            GROUP BY sources.id
            ORDER BY sources.createdAt DESC
        """).fetchall()
        return rowsToList(rows)
    finally:
        connection.close()


def fetchDueQuestions(connection, limit):
    rows = connection.execute(
        questionWithSourceQuery + """
        WHERE questions.status = 'approved'
          AND questions.paused = 0
          AND (questions.nextReview IS NULL OR questions.nextReview <= ?)
        ORDER BY questions.nextReview IS NULL DESC, questions.nextReview ASC
        LIMIT ?
        """,
        (utcNowText(), limit),
    ).fetchall()
    return rowsToList(rows)


@app.get("/api/due")
def getDueQuestions(limit: int = 20):
    limit = max(1, min(limit, 100))
    connection = getDatabase()
    try:
        return fetchDueQuestions(connection, limit)
    finally:
        connection.close()


@app.post("/api/review")
async def submitReview(request: Request):
    body, error = await readJsonBody(request)
    if error:
        return error

    questionId = body.get("questionId")
    quality = body.get("quality")

    if not isinstance(questionId, int):
        return errorResponse("questionId must be an integer", 400)
    if not isinstance(quality, int) or quality < 0 or quality > 5:
        return errorResponse("quality must be an integer 0-5", 400)

    connection = getDatabase()
    try:
        question = connection.execute("SELECT id, status FROM questions WHERE id = ?", (questionId,)).fetchone()
        if not question:
            return errorResponse("Question not found", 404)
        if question["status"] != "approved":
            return errorResponse("Only approved questions can be reviewed", 400)

        result = gradeQuestion(connection, questionId, quality)
        if not result:
            return errorResponse("Failed to update", 500)

        return {"questionId": questionId, "quality": quality, **result}
    finally:
        connection.close()


@app.get("/api/stats")
def getStudyStats():
    connection = getDatabase()
    try:
        now = utcNowText()
        oneWeekLater = (utcNow() + timedelta(days=7)).isoformat()

        row = connection.execute("""
            SELECT
                SUM(CASE WHEN paused = 0 AND (nextReview IS NULL OR nextReview <= :now) THEN 1 ELSE 0 END) AS dueNow,
                COUNT(*) AS totalApproved,
                SUM(CASE WHEN repetitions > 0 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN paused = 0 AND nextReview > :now AND nextReview <= :week THEN 1 ELSE 0 END) AS upcomingWeek,
                SUM(CASE WHEN paused = 1 THEN 1 ELSE 0 END) AS pausedCount
            FROM questions
            WHERE status = 'approved'
        """, {"now": now, "week": oneWeekLater}).fetchone()

        return {
            "dueNow": row["dueNow"] or 0,
            "totalInRotation": row["totalApproved"] or 0,
            "reviewedAtLeastOnce": row["reviewed"] or 0,
            "upcomingSevenDays": row["upcomingWeek"] or 0,
            "paused": row["pausedCount"] or 0,
        }
    finally:
        connection.close()


@app.get("/api/counts")
def getCounts():
    connection = getDatabase()
    try:
        row = connection.execute(f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) AS approved,
                SUM(CASE WHEN {dueCondition} THEN 1 ELSE 0 END) AS due
            FROM questions
        """, (utcNowText(),)).fetchone()

        return {
            "total": row["total"] or 0,
            "pending": row["pending"] or 0,
            "approved": row["approved"] or 0,
            "due": row["due"] or 0,
        }
    finally:
        connection.close()


@app.get("/api/upcoming")
def getUpcomingReviews():
    connection = getDatabase()
    try:
        rows = connection.execute("""
            SELECT questions.id, questions.question, questions.nextReview, questions.interval,
                   sources.title AS sourceTitle
            FROM questions
            LEFT JOIN sources ON questions.sourceId = sources.id
            WHERE questions.status = 'approved'
              AND questions.nextReview IS NOT NULL
              AND questions.nextReview > ?
            ORDER BY questions.nextReview ASC
            LIMIT 500
        """, (utcNowText(),)).fetchall()
        return rowsToList(rows)
    finally:
        connection.close()


def pauseAllQuestions():
    connection = getDatabase()
    try:
        result = connection.execute(
            "UPDATE questions SET paused = 1, nextReview = NULL WHERE status = 'approved' AND paused = 0"
        )
        connection.commit()
        return result.rowcount
    finally:
        connection.close()


def resumeAllQuestions():
    connection = getDatabase()
    try:
        result = connection.execute(
            "UPDATE questions SET paused = 0, nextReview = ? WHERE status = 'approved' AND paused = 1",
            (utcNowText(),),
        )
        connection.commit()
        return result.rowcount
    finally:
        connection.close()


@app.post("/api/pause")
def pauseSchedule():
    return {"paused": pauseAllQuestions()}


@app.post("/api/resume")
def resumeSchedule():
    return {"resumed": resumeAllQuestions()}


@app.get("/", response_class=HTMLResponse)
async def homePage():
    return htmlPage


telegramApp = None
studySessions = {}

gradeLabels = {1: "Don't Know Yet", 3: "Revise", 5: "Easy"}
repeatsPerGrade = {5: 1, 3: 2, 1: 3}


def plural(count):
    if count == 1:
        return ""
    return "s"


def escapeHtml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def sendQuestion(context, question: dict, targetChatId: str = None, remaining: int = None):
    messageText = f"<b>Question:</b>\n{escapeHtml(question['question'])}"
    if remaining is not None:
        messageText += f"\n\n<i>({remaining} remaining)</i>"

    keyboard = [[InlineKeyboardButton("Show Answer", callback_data=f"show_{question['id']}")]]

    await context.bot.send_message(
        chat_id=targetChatId or chatId,
        text=messageText,
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def getDueQuestionsForBot(limit=1):
    connection = getDatabase()
    try:
        return fetchDueQuestions(connection, limit)
    finally:
        connection.close()


def countDueQuestions():
    connection = getDatabase()
    try:
        row = connection.execute(f"SELECT COUNT(*) FROM questions WHERE {dueCondition}", (utcNowText(),)).fetchone()
        return row[0]
    finally:
        connection.close()


def gradeQuestionFromBot(questionId, quality):
    connection = getDatabase()
    try:
        row = connection.execute("SELECT status FROM questions WHERE id = ?", (questionId,)).fetchone()
        if not row or row["status"] != "approved":
            return None
        result = gradeQuestion(connection, questionId, quality)
        return result["interval"]
    finally:
        connection.close()


async def startCommand(update, context):
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


async def studyCommand(update, context):
    currentChatId = str(update.effective_chat.id)

    if studySessions.get(currentChatId):
        await sendNextQuestion(context, currentChatId)
        return

    dueQuestions = getDueQuestionsForBot(limit=200)
    if not dueQuestions:
        await update.message.reply_text("No questions due right now. You're caught up!")
        return

    questionIds = []
    for question in dueQuestions:
        questionIds.append(question["id"])

    studySessions[currentChatId] = {
        "queue": questionIds,
        "repeatQueue": [],
        "total": len(dueQuestions),
        "answered": 0,
        "questionProgress": {},
    }

    count = len(dueQuestions)
    await update.message.reply_text(f"Starting study session — {count} question{plural(count)} due. /pause to stop.")
    await sendNextQuestion(context, currentChatId)


async def finishSession(context, currentChatId):
    total = studySessions[currentChatId]["total"]
    del studySessions[currentChatId]
    await context.bot.send_message(
        chat_id=currentChatId,
        text=f"Session complete! Reviewed {total} question{plural(total)}. You're caught up!",
    )


def takeNextQuestion(session):
    connection = getDatabase()
    try:
        while session["queue"] or session["repeatQueue"]:
            if not session["queue"]:
                session["queue"] = session["repeatQueue"]
                session["repeatQueue"] = []

            questionId = session["queue"].pop(0)
            row = connection.execute(
                "SELECT id, question, answer, type, easinessFactor, interval, repetitions FROM questions WHERE id = ?",
                (questionId,),
            ).fetchone()
            if row:
                return dict(row)
        return None
    finally:
        connection.close()


async def sendNextQuestion(context, currentChatId: str):
    session = studySessions.get(currentChatId)
    if not session:
        return

    if not session["queue"] and not session["repeatQueue"]:
        await finishSession(context, currentChatId)
        return

    question = takeNextQuestion(session)
    if question is None:
        await finishSession(context, currentChatId)
        return

    remaining = len(session["queue"]) + len(session["repeatQueue"])
    await sendQuestion(context, question, targetChatId=currentChatId, remaining=remaining)


async def dueCommand(update, context):
    count = countDueQuestions()
    if count == 0:
        await update.message.reply_text("No questions due. You're caught up.")
    else:
        await update.message.reply_text(f"{count} question{plural(count)} due.")


async def statsCommand(update, context):
    connection = getDatabase()
    try:
        due = connection.execute(f"SELECT COUNT(*) FROM questions WHERE {dueCondition}", (utcNowText(),)).fetchone()[0]
        total = connection.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved'").fetchone()[0]
        reviewed = connection.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved' AND repetitions > 0").fetchone()[0]
        paused = connection.execute("SELECT COUNT(*) FROM questions WHERE status = 'approved' AND paused = 1").fetchone()[0]
    finally:
        connection.close()

    await update.message.reply_text(
        f"Due now: {due}\nIn rotation: {total}\nReviewed at least once: {reviewed}\nPaused: {paused}"
    )


async def resetCommand(update, context):
    currentChatId = str(update.effective_chat.id)

    connection = getDatabase()
    try:
        result = connection.execute("""
            UPDATE questions SET easinessFactor = 2.5, interval = 0, repetitions = 0, nextReview = ?
            WHERE status = 'approved' AND paused = 0
        """, (utcNowText(),))
        connection.commit()
        count = result.rowcount
    finally:
        connection.close()

    studySessions.pop(currentChatId, None)
    await update.message.reply_text(f"Reset {count} questions to fresh. All due now — /study to start.")


async def pauseCommand(update, context):
    currentChatId = str(update.effective_chat.id)
    session = studySessions.pop(currentChatId, None)

    if session:
        answered = session["answered"]
        await update.message.reply_text(f"Study paused. Answered {answered} question{plural(answered)} this session.")
    else:
        await update.message.reply_text("No active study session.")


async def muteCommand(update, context):
    count = pauseAllQuestions()
    studySessions.pop(str(update.effective_chat.id), None)
    await update.message.reply_text(f"Muted {count} questions. No notifications until you /unmute.")


async def unmuteCommand(update, context):
    count = resumeAllQuestions()
    await update.message.reply_text(f"Unmuted {count} questions. They're all due now — /study to start.")


async def showAnswer(buttonPress, questionId):
    connection = getDatabase()
    try:
        row = connection.execute("SELECT question, answer, type FROM questions WHERE id = ?", (questionId,)).fetchone()
    finally:
        connection.close()

    if not row:
        await buttonPress.edit_message_text("Question not found (may have been deleted).")
        return

    messageText = f"<b>Question:</b>\n{escapeHtml(row['question'])}\n\n<b>Answer:</b>\n{escapeHtml(row['answer'])}"
    keyboard = [[
        InlineKeyboardButton("✅ Easy", callback_data=f"grade_{questionId}_5"),
        InlineKeyboardButton("🔄 Revise", callback_data=f"grade_{questionId}_3"),
        InlineKeyboardButton("❌ Don't Know Yet", callback_data=f"grade_{questionId}_1"),
    ]]
    await buttonPress.edit_message_text(text=messageText, parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))


async def gradeAndReport(buttonPress, questionId, quality):
    newInterval = gradeQuestionFromBot(questionId, quality)
    if newInterval is None:
        await buttonPress.edit_message_text("Failed to grade — question may have been deleted.")
        return False

    label = gradeLabels.get(quality, str(quality))
    await buttonPress.edit_message_text(
        f"<b>{label}</b> — next review in {newInterval} day{plural(newInterval)}.",
        parse_mode="HTML",
    )
    return True


async def handleGradeInSession(context, buttonPress, session, currentChatId, questionId, quality):
    progress = session["questionProgress"]

    if questionId not in progress:
        progress[questionId] = {
            "repeatsLeft": repeatsPerGrade.get(quality, 1),
            "lastQuality": quality,
        }
    else:
        progress[questionId]["repeatsLeft"] -= 1
        progress[questionId]["lastQuality"] = quality

    questionProgress = progress[questionId]
    session["answered"] += 1

    if questionProgress["repeatsLeft"] > 0:
        session["repeatQueue"].append(questionId)
        repeatsLeft = questionProgress["repeatsLeft"]
        label = gradeLabels.get(quality, str(quality))
        await buttonPress.edit_message_text(
            f"<b>{label}</b> — will come back {repeatsLeft} more time{plural(repeatsLeft)}.",
            parse_mode="HTML",
        )
        await sendNextQuestion(context, currentChatId)
        return

    wasGraded = await gradeAndReport(buttonPress, questionId, questionProgress["lastQuality"])
    if not wasGraded:
        return

    del progress[questionId]
    await sendNextQuestion(context, currentChatId)


async def handleButtonPress(update, context):
    buttonPress = update.callback_query
    await buttonPress.answer()

    buttonData = buttonPress.data
    currentChatId = str(buttonPress.message.chat_id)

    if buttonData.startswith("show_"):
        questionId = int(buttonData.split("_")[1])
        await showAnswer(buttonPress, questionId)

    elif buttonData.startswith("grade_"):
        parts = buttonData.split("_")
        questionId = int(parts[1])
        quality = int(parts[2])

        session = studySessions.get(currentChatId)
        if session:
            await handleGradeInSession(context, buttonPress, session, currentChatId, questionId, quality)
        else:
            await gradeAndReport(buttonPress, questionId, quality)


def countDueQuestionsByTopic():
    connection = getDatabase()
    try:
        rows = connection.execute("""
            SELECT COALESCE(sources.topicName, sources.title, 'Unknown') AS topic, COUNT(*) AS questionCount
            FROM questions
            LEFT JOIN sources ON questions.sourceId = sources.id
            WHERE questions.status = 'approved' AND questions.paused = 0
              AND (questions.nextReview IS NULL OR questions.nextReview <= ?)
            GROUP BY topic
            ORDER BY questionCount DESC
        """, (utcNowText(),)).fetchall()

        topicCounts = []
        for row in rows:
            topicCounts.append((row["topic"], row["questionCount"]))
        return topicCounts
    finally:
        connection.close()


async def sendScheduledNudge(context):
    localTime = datetime.now().astimezone()
    if localTime.hour < studyHoursStart or localTime.hour >= studyHoursEnd:
        logger.info(f"Outside study hours ({studyHoursStart}-{studyHoursEnd}). Skipping.")
        return

    if studySessions.get(chatId):
        logger.info("User is in active study session. Skipping nudge.")
        return

    topicCounts = countDueQuestionsByTopic()
    if not topicCounts:
        logger.info("No due questions.")
        return

    total = 0
    for topic, questionCount in topicCounts:
        total += questionCount

    lines = [f"You have {total} question{plural(total)} due:"]
    for topic, questionCount in topicCounts:
        lines.append(f"  • {topic}: {questionCount}")
    lines.append("\n/study to start.")

    await context.bot.send_message(chat_id=chatId, text="\n".join(lines))
    logger.info(f"Sent nudge: {total} due across {len(topicCounts)} topics.")


async def startTelegramBot():
    global telegramApp
    if not telegramEnabled:
        logger.info("Telegram bot not enabled. Skipping.")
        return

    logger.info("Starting embedded Telegram bot...")
    telegramApp = Application.builder().token(botToken).build()

    telegramApp.add_handler(CommandHandler("start", startCommand))
    telegramApp.add_handler(CommandHandler("study", studyCommand))
    telegramApp.add_handler(CommandHandler("due", dueCommand))
    telegramApp.add_handler(CommandHandler("stats", statsCommand))
    telegramApp.add_handler(CommandHandler("pause", pauseCommand))
    telegramApp.add_handler(CommandHandler("reset", resetCommand))
    telegramApp.add_handler(CommandHandler("mute", muteCommand))
    telegramApp.add_handler(CommandHandler("unmute", unmuteCommand))
    telegramApp.add_handler(CallbackQueryHandler(handleButtonPress))

    telegramApp.job_queue.run_repeating(sendScheduledNudge, interval=nudgeIntervalSeconds, first=10)

    await telegramApp.initialize()
    await telegramApp.start()
    await telegramApp.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram bot started.")


async def stopTelegramBot():
    global telegramApp
    if telegramApp is None:
        return

    logger.info("Stopping Telegram bot...")
    await telegramApp.updater.stop()
    await telegramApp.stop()
    await telegramApp.shutdown()
    telegramApp = None
    logger.info("Telegram bot stopped.")


htmlPage = """\
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

        .source-label {
            font-size: 0.72rem;
            color: #556;
        }

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

        <div id="view-review" class="view">
            <div id="reviewBatchBar" class="batch-bar" style="display:none;">
                <input type="checkbox" id="selectAllPending" onchange="toggleSelectAll('pending')">
                <span class="count" id="reviewSelCount">0 selected</span>
                <button class="btn btn-approve btn-sm" onclick="batchAction('approve', 'pending')">Approve Selected</button>
                <button class="btn btn-delete btn-sm" onclick="batchAction('delete', 'pending')">Delete Selected</button>
            </div>
            <div id="reviewList"></div>
        </div>

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
        const maxCharacters = 100000;
        let currentView = 'generate';
        let currentSubtab = 'text';
        let selectedFile = null;
        let bankFilter = 'all';

        const selectedIds = { pending: new Set(), bank: new Set() };

        function getElement(id) {
            return document.getElementById(id);
        }

        function makeElement(tagName, className, text) {
            const element = document.createElement(tagName);
            if (className) element.className = className;
            if (text !== undefined) element.textContent = text;
            return element;
        }

        async function getJson(url) {
            const response = await fetch(url);
            return response.json();
        }

        function sendJson(url, method, data) {
            return fetch(url, {
                method: method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
        }

        function showStatBoxes(container, stats) {
            container.innerHTML = '';
            stats.forEach(stat => {
                const box = makeElement('div', 'stat-box');
                box.appendChild(makeElement('div', 'num', stat.number));
                box.appendChild(makeElement('div', 'label', stat.label));
                container.appendChild(box);
            });
        }

        function switchView(view) {
            currentView = view;

            const navButtons = document.querySelectorAll('.main-nav button');
            navButtons.forEach(button => button.classList.remove('active'));
            document.querySelectorAll('.view').forEach(section => section.classList.remove('active'));

            getElement('view-' + view).classList.add('active');
            const position = ['generate', 'review', 'study', 'bank'].indexOf(view);
            navButtons[position].classList.add('active');

            if (view === 'review') loadReview();
            if (view === 'study') loadStudy();
            if (view === 'bank') loadBank();
        }

        function switchSubtab(tab) {
            currentSubtab = tab;

            const tabButtons = document.querySelectorAll('.subtab');
            tabButtons.forEach(button => button.classList.remove('active'));
            document.querySelectorAll('.subtab-content').forEach(content => content.classList.remove('active'));

            getElement('subtab-' + tab).classList.add('active');
            if (tab === 'text') {
                tabButtons[0].classList.add('active');
            } else {
                tabButtons[1].classList.add('active');
            }
        }

        getElement('source').addEventListener('input', function() {
            const length = this.value.length;
            const counter = getElement('charCount');
            counter.textContent = length.toLocaleString() + ' / 100,000';
            counter.classList.toggle('warn', length > maxCharacters * 0.9);
        });

        const dropZone = getElement('dropZone');
        const fileInput = getElement('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (event) => {
            event.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

        dropZone.addEventListener('drop', (event) => {
            event.preventDefault();
            dropZone.classList.remove('drag-over');
            const droppedFile = event.dataTransfer.files[0];
            if (droppedFile) selectPdf(droppedFile);
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) selectPdf(fileInput.files[0]);
        });

        function selectPdf(file) {
            const status = getElement('genStatus');

            if (file.type !== 'application/pdf') {
                status.textContent = 'Error: Only PDF files.';
                return;
            }
            if (file.size > 20 * 1024 * 1024) {
                status.textContent = 'Error: PDF exceeds 20MB.';
                return;
            }

            selectedFile = file;
            const sizeInKb = (file.size / 1024).toFixed(0);
            getElement('fileName').textContent = file.name + ' (' + sizeInKb + ' KB)';

            const titleInput = getElement('sourceTitle');
            if (!titleInput.value.trim()) {
                titleInput.value = file.name.replace(/\\.pdf$/i, '');
            }
            status.textContent = '';
        }

        function clearGenerateInputs() {
            getElement('source').value = '';
            getElement('charCount').textContent = '0 / 100,000';
            getElement('sourceTitle').value = '';
            selectedFile = null;
            getElement('fileName').textContent = '';
            fileInput.value = '';
        }

        async function generate() {
            const generateButton = getElement('genBtn');
            const status = getElement('genStatus');
            const title = getElement('sourceTitle').value.trim();

            generateButton.disabled = true;
            generateButton.textContent = 'Generating...';

            try {
                let response;

                if (currentSubtab === 'text') {
                    const text = getElement('source').value.trim();
                    if (!text) {
                        status.textContent = 'Paste some text first.';
                        return;
                    }
                    status.textContent = 'Sending to Groq API...';
                    response = await sendJson('/generate', 'POST', { text: text, title: title || 'Untitled' });
                } else {
                    if (!selectedFile) {
                        status.textContent = 'Select a PDF first.';
                        return;
                    }
                    status.textContent = 'Uploading PDF and extracting text...';
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    response = await fetch('/upload-pdf', { method: 'POST', body: formData });
                }

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Request failed');
                }

                const result = await response.json();
                const questionCount = result.questions ? result.questions.length : 0;

                if (questionCount === 0) {
                    status.textContent = 'No questions generated. Try different material.';
                    return;
                }

                let message = questionCount + ' questions generated from ' + result.chunkCount + ' chunk(s).';
                if (result.extractedChars) {
                    message += ' ' + result.extractedChars.toLocaleString() + ' chars extracted.';
                }
                message += ' Go to Review to vet them.';
                status.textContent = message;

                clearGenerateInputs();
                updateBadges();
            } catch (error) {
                status.textContent = 'Error: ' + error.message;
            } finally {
                generateButton.disabled = false;
                generateButton.textContent = 'Generate Questions';
            }
        }

        function showQuestionCards(list, questions, context) {
            const fragment = document.createDocumentFragment();
            questions.forEach(question => fragment.appendChild(makeCard(question, context)));
            list.appendChild(fragment);
        }

        async function loadReview() {
            const list = getElement('reviewList');
            list.innerHTML = '';
            selectedIds.pending.clear();
            updateSelectionCount('pending');
            getElement('selectAllPending').checked = false;

            try {
                const questions = await getJson('/api/questions?status=pending');
                getElement('pendingBadge').textContent = questions.length;

                if (questions.length === 0) {
                    list.innerHTML = '<div class="empty-state">No pending questions. Generate some from the Generate tab.</div>';
                    getElement('reviewBatchBar').style.display = 'none';
                    return;
                }

                getElement('reviewBatchBar').style.display = 'flex';
                showQuestionCards(list, questions, 'pending');
            } catch (error) {
                list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
            }
        }

        async function loadBank() {
            const list = getElement('bankList');
            list.innerHTML = '';
            selectedIds.bank.clear();
            updateSelectionCount('bank');
            getElement('selectAllBank').checked = false;

            try {
                let questionsUrl = '/api/questions';
                if (bankFilter !== 'all') {
                    questionsUrl += '?status=' + bankFilter;
                }

                const [questions, counts] = await Promise.all([
                    getJson(questionsUrl),
                    getJson('/api/counts'),
                ]);

                getElement('approvedBadge').textContent = counts.approved;
                getElement('pendingBadge').textContent = counts.pending;

                showStatBoxes(getElement('bankStats'), [
                    { number: counts.total, label: 'Total' },
                    { number: counts.approved, label: 'Approved' },
                    { number: counts.pending, label: 'Pending' },
                ]);

                if (questions.length === 0) {
                    list.innerHTML = '<div class="empty-state">No questions found.</div>';
                    getElement('bankBatchBar').style.display = 'none';
                    return;
                }

                getElement('bankBatchBar').style.display = 'flex';
                showQuestionCards(list, questions, 'bank');
            } catch (error) {
                list.innerHTML = '<div class="empty-state">Failed to load questions.</div>';
            }
        }

        function setBankFilter(filter, button) {
            bankFilter = filter;
            document.querySelectorAll('.filter-btn').forEach(filterButton => filterButton.classList.remove('active'));
            button.classList.add('active');
            loadBank();
        }

        function makeButton(className, text, onClick) {
            const button = makeElement('button', className, text);
            button.onclick = onClick;
            return button;
        }

        function makeCard(question, context) {
            const card = makeElement('div', 'question-card');
            card.dataset.id = question.id;
            card.id = 'card-' + question.id;

            const header = makeElement('div', 'card-header');

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.onchange = () => toggleSelect(question.id, context);
            if (selectedIds[context].has(question.id)) checkbox.checked = true;

            const body = makeElement('div', 'card-body');
            const meta = makeElement('div', 'meta');

            meta.appendChild(makeElement('span', 'tag', question.type || 'unknown'));

            if (question.status) {
                meta.appendChild(makeElement('span', 'tag ' + question.status, question.status));
            }

            if (question.sourceTitle) {
                meta.appendChild(makeElement('span', 'source-label', question.sourceTitle));
            }

            const actions = makeElement('div', 'actions');

            if (question.status === 'pending') {
                actions.appendChild(makeButton('btn btn-approve btn-sm', 'Approve',
                    () => updateStatus(question.id, 'approved', context)));
            }
            actions.appendChild(makeButton('btn btn-edit btn-sm', 'Edit',
                () => startEdit(question.id, question.question, question.answer)));
            actions.appendChild(makeButton('btn btn-delete btn-sm', 'Delete',
                () => deleteQuestion(question.id, context)));

            body.appendChild(makeElement('div', 'q', question.question));
            body.appendChild(makeElement('div', 'a', question.answer));
            body.appendChild(meta);
            body.appendChild(actions);

            header.appendChild(checkbox);
            header.appendChild(body);
            card.appendChild(header);

            return card;
        }

        function reloadCurrentList() {
            if (currentView === 'review') {
                loadReview();
            } else {
                loadBank();
            }
        }

        function removeCard(id, context) {
            const card = getElement('card-' + id);
            if (card) card.remove();
            selectedIds[context].delete(id);
            updateSelectionCount(context);
        }

        async function updateStatus(id, status, context) {
            try {
                const response = await sendJson('/api/questions/' + id, 'PUT', { status: status });
                if (!response.ok) throw new Error('Failed');

                if (context === 'pending') {
                    removeCard(id, context);
                } else {
                    loadBank();
                }
                updateBadges();
            } catch (error) {
                console.error(error);
            }
        }

        async function deleteQuestion(id, context) {
            try {
                const response = await fetch('/api/questions/' + id, { method: 'DELETE' });
                if (!response.ok) throw new Error('Failed');
                removeCard(id, context);
                updateBadges();
            } catch (error) {
                console.error(error);
            }
        }

        function startEdit(id, question, answer) {
            const card = getElement('card-' + id);
            const body = card.querySelector('.card-body');
            body.innerHTML = '';

            const questionInput = makeElement('input', 'edit-field');
            questionInput.type = 'text';
            questionInput.value = question;
            questionInput.placeholder = 'Question';

            const answerInput = makeElement('textarea', 'edit-field');
            answerInput.value = answer;
            answerInput.placeholder = 'Answer';

            const buttonRow = makeElement('div', 'actions');
            buttonRow.appendChild(makeButton('btn btn-primary btn-sm', 'Save',
                () => saveEdit(id, questionInput.value, answerInput.value)));
            buttonRow.appendChild(makeButton('btn btn-ghost btn-sm', 'Cancel', reloadCurrentList));

            body.appendChild(questionInput);
            body.appendChild(answerInput);
            body.appendChild(buttonRow);
            questionInput.focus();
        }

        async function saveEdit(id, question, answer) {
            question = question.trim();
            answer = answer.trim();
            if (!question || !answer) return;

            try {
                const response = await sendJson('/api/questions/' + id, 'PUT', { question: question, answer: answer });
                if (!response.ok) throw new Error('Failed');
                reloadCurrentList();
            } catch (error) {
                console.error(error);
            }
        }

        function toggleSelect(id, context) {
            if (selectedIds[context].has(id)) {
                selectedIds[context].delete(id);
            } else {
                selectedIds[context].add(id);
            }
            updateSelectionCount(context);

            const card = getElement('card-' + id);
            if (card) card.classList.toggle('selected', selectedIds[context].has(id));
        }

        function toggleSelectAll(context) {
            const checkboxId = context === 'pending' ? 'selectAllPending' : 'selectAllBank';
            const listId = context === 'pending' ? 'reviewList' : 'bankList';
            const selectAll = getElement(checkboxId).checked;

            const cards = getElement(listId).querySelectorAll('.question-card');
            cards.forEach(card => {
                const id = parseInt(card.dataset.id);
                const checkbox = card.querySelector('input[type="checkbox"]');

                if (selectAll) {
                    selectedIds[context].add(id);
                    card.classList.add('selected');
                } else {
                    selectedIds[context].delete(id);
                    card.classList.remove('selected');
                }
                if (checkbox) checkbox.checked = selectAll;
            });
            updateSelectionCount(context);
        }

        function updateSelectionCount(context) {
            const countId = context === 'pending' ? 'reviewSelCount' : 'bankSelCount';
            getElement(countId).textContent = selectedIds[context].size + ' selected';
        }

        async function batchAction(action, context) {
            const ids = Array.from(selectedIds[context]);
            if (ids.length === 0) return;

            try {
                const response = await sendJson('/api/questions/batch', 'POST', { ids: ids, action: action });
                if (!response.ok) throw new Error('Failed');

                selectedIds[context].clear();
                if (context === 'pending') {
                    loadReview();
                } else {
                    loadBank();
                }
                updateBadges();
            } catch (error) {
                console.error(error);
            }
        }

        let studyQueue = [];
        let studyPosition = 0;
        let studyReviewedCount = 0;
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
            } catch (error) {
                console.error('Pause toggle failed:', error);
            }
        }

        function updatePauseButton(stats) {
            const bar = getElement('pauseBar');
            const button = getElement('pauseBtn');
            const status = getElement('pauseStatus');

            const hasQuestionsInRotation = stats.totalInRotation > 0;
            if (!hasQuestionsInRotation) {
                bar.style.display = 'none';
                return;
            }

            bar.style.display = 'flex';
            bar.style.alignItems = 'center';
            bar.style.gap = '0.8rem';

            isPaused = stats.paused > 0;
            if (isPaused) {
                button.textContent = 'Resume Schedule';
                button.classList.add('paused');
                status.textContent = stats.paused + ' questions paused';
            } else {
                button.textContent = 'Pause Schedule';
                button.classList.remove('paused');
                status.textContent = '';
            }
        }

        async function loadStudy() {
            const emptyMessage = getElement('studyEmpty');
            const sessionSection = getElement('studySession');
            const doneSection = getElement('studyDone');

            emptyMessage.style.display = 'none';
            sessionSection.style.display = 'none';
            doneSection.style.display = 'none';

            try {
                const [stats, dueQuestions] = await Promise.all([
                    getJson('/api/stats'),
                    getJson('/api/due'),
                ]);

                showStatBoxes(getElement('studyStats'), [
                    { number: stats.dueNow, label: 'Due Now' },
                    { number: stats.totalInRotation, label: 'In Rotation' },
                    { number: stats.reviewedAtLeastOnce, label: 'Reviewed' },
                    { number: stats.upcomingSevenDays, label: 'Next 7 Days' },
                ]);

                updatePauseButton(stats);
                getElement('dueBadge').textContent = stats.dueNow;

                if (dueQuestions.length === 0) {
                    emptyMessage.style.display = 'block';
                    emptyMessage.textContent = 'No questions due right now. Check back later or approve more questions.';
                    loadUpcoming();
                    return;
                }

                studyQueue = dueQuestions;
                studyPosition = 0;
                studyReviewedCount = 0;
                sessionSection.style.display = 'block';
                showCard();
                loadUpcoming();
            } catch (error) {
                emptyMessage.style.display = 'block';
                emptyMessage.textContent = 'Failed to load study session.';
            }
        }

        function showCard() {
            if (studyPosition >= studyQueue.length) {
                getElement('studySession').style.display = 'none';
                getElement('studyDone').style.display = 'block';
                getElement('studyDoneCount').textContent = studyReviewedCount;
                updateBadges();
                loadUpcoming();
                return;
            }

            const question = studyQueue[studyPosition];
            cardFlipped = false;

            getElement('studyProgress').textContent = (studyPosition + 1) + ' / ' + studyQueue.length;
            getElement('cardQuestion').textContent = question.question;
            getElement('cardAnswer').textContent = question.answer;
            getElement('cardSource').textContent = question.sourceTitle || '';

            getElement('cardFront').style.display = 'block';
            getElement('cardBack').style.display = 'none';
            getElement('gradeButtons').style.display = 'none';
        }

        function flipCard() {
            if (cardFlipped) return;
            cardFlipped = true;
            getElement('cardFront').style.display = 'none';
            getElement('cardBack').style.display = 'block';
            getElement('gradeButtons').style.display = 'block';
        }

        function submitGrade(quality) {
            const question = studyQueue[studyPosition];

            sendJson('/api/review', 'POST', { questionId: question.id, quality: quality })
                .catch(error => console.error('Failed to submit grade:', error));

            studyReviewedCount++;
            studyPosition++;
            showCard();
        }

        function getDayLabel(reviewDate, now) {
            const oneDay = 1000 * 60 * 60 * 24;
            const daysAway = Math.ceil((reviewDate - now) / oneDay);

            if (daysAway <= 0) return 'Today';
            if (daysAway === 1) return 'Tomorrow';
            return reviewDate.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
        }

        function groupByDay(questions) {
            const groups = {};
            const now = new Date();

            questions.forEach(question => {
                const label = getDayLabel(new Date(question.nextReview), now);
                if (!groups[label]) groups[label] = [];
                groups[label].push(question);
            });

            return groups;
        }

        function makeUpcomingItem(question) {
            const item = makeElement('div', 'upcoming-item');
            const meta = makeElement('div', 'upcoming-meta');

            if (question.sourceTitle) {
                meta.appendChild(makeElement('span', 'upcoming-interval', question.sourceTitle));
            }
            meta.appendChild(makeElement('span', 'upcoming-interval', question.interval + 'd interval'));

            item.appendChild(makeElement('div', 'upcoming-q', question.question));
            item.appendChild(meta);
            return item;
        }

        async function loadUpcoming() {
            const list = getElement('upcomingList');
            const section = getElement('upcomingSection');
            list.innerHTML = '';

            try {
                const upcoming = await getJson('/api/upcoming');

                if (upcoming.length === 0) {
                    section.style.display = 'none';
                    return;
                }
                section.style.display = 'block';

                const groups = groupByDay(upcoming);
                const fragment = document.createDocumentFragment();

                for (const [day, questions] of Object.entries(groups)) {
                    const dayBlock = makeElement('div', 'upcoming-day');
                    dayBlock.appendChild(makeElement('div', 'upcoming-day-header', day + ' (' + questions.length + ')'));
                    questions.forEach(question => dayBlock.appendChild(makeUpcomingItem(question)));
                    fragment.appendChild(dayBlock);
                }

                list.appendChild(fragment);
            } catch (error) {
                section.style.display = 'none';
            }
        }

        async function updateBadges() {
            try {
                const counts = await getJson('/api/counts');
                getElement('pendingBadge').textContent = counts.pending;
                getElement('approvedBadge').textContent = counts.approved;
                getElement('dueBadge').textContent = counts.due;
            } catch (error) {
            }
        }

        updateBadges();

        document.addEventListener('keydown', function(event) {
            if (currentView !== 'study') return;
            if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;

            if (event.code === 'Space' && !cardFlipped) {
                event.preventDefault();
                flipCard();
            } else if (cardFlipped && event.key >= '0' && event.key <= '5') {
                event.preventDefault();
                submitGrade(parseInt(event.key));
            }
        });
    </script>
</body>
</html>
"""
