import asyncio
import json
from contextlib import asynccontextmanager, closing
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from config import (
    allowedQuestionTypes,
    allowedStatuses,
    maxInputCharacters,
    maxJsonBodyBytes,
    maxPdfBytes,
)
from database import (
    createTables,
    dueCondition,
    fetchDueQuestions,
    getDatabase,
    pauseAllQuestions,
    questionWithSourceQuery,
    readStudyStats,
    resumeAllQuestions,
    rowsToList,
    utcNowText,
)
from questionGenerator import extractPdfText, generateQuestionsFromText
from spacedRepetition import gradeQuestion
from telegramBot import startTelegramBot, stopTelegramBot


@asynccontextmanager
async def lifespan(fastApiApp: FastAPI):
    await startTelegramBot()
    yield
    await stopTelegramBot()


app = FastAPI(lifespan=lifespan)

createTables()


staticFolder = Path(__file__).parent / "static"


def readStaticFile(fileName):
    return (staticFolder / fileName).read_text(encoding="utf-8")


pageHtml = readStaticFile("index.html")
pageCss = readStaticFile("style.css")
pageScript = readStaticFile("script.js")


@app.get("/", response_class=HTMLResponse)
async def homePage():
    return pageHtml


@app.get("/style.css")
async def styleFile():
    return Response(pageCss, media_type="text/css")


@app.get("/script.js")
async def scriptFile():
    return Response(pageScript, media_type="text/javascript")


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


@app.get("/api/questions")
def listQuestions(status: str = None, sourceId: int = None):
    with closing(getDatabase()) as connection:
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

    with closing(getDatabase()) as connection:
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


@app.delete("/api/questions/{questionId}")
def deleteQuestion(questionId: int):
    with closing(getDatabase()) as connection:
        if not questionExists(connection, questionId):
            return errorResponse("Question not found", 404)

        connection.execute("DELETE FROM questions WHERE id = ?", (questionId,))
        connection.commit()
        return {"deleted": questionId}


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

    with closing(getDatabase()) as connection:
        if action == "approve":
            connection.execute(
                f"UPDATE questions SET status = 'approved', nextReview = COALESCE(nextReview, ?) WHERE id IN ({placeholders})",
                [utcNowText()] + ids,
            )
        else:
            connection.execute(f"DELETE FROM questions WHERE id IN ({placeholders})", ids)

        connection.commit()
        return {"action": action, "count": len(ids)}


@app.get("/api/sources")
def listSources():
    with closing(getDatabase()) as connection:
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


@app.get("/api/due")
def getDueQuestions(limit: int = 20):
    limit = max(1, min(limit, 100))
    with closing(getDatabase()) as connection:
        return fetchDueQuestions(connection, limit)


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

    with closing(getDatabase()) as connection:
        question = connection.execute("SELECT id, status FROM questions WHERE id = ?", (questionId,)).fetchone()
        if not question:
            return errorResponse("Question not found", 404)
        if question["status"] != "approved":
            return errorResponse("Only approved questions can be reviewed", 400)

        result = gradeQuestion(connection, questionId, quality)
        if not result:
            return errorResponse("Failed to update", 500)

        return {"questionId": questionId, "quality": quality, **result}


@app.get("/api/stats")
def getStudyStats():
    return readStudyStats()


@app.get("/api/counts")
def getCounts():
    with closing(getDatabase()) as connection:
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


@app.get("/api/upcoming")
def getUpcomingReviews():
    with closing(getDatabase()) as connection:
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


@app.post("/api/pause")
def pauseSchedule():
    return {"paused": pauseAllQuestions()}


@app.post("/api/resume")
def resumeSchedule():
    return {"resumed": resumeAllQuestions()}
