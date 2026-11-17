import asyncio
import json
import time
from contextlib import closing
from pathlib import Path

import fitz
from groq import Groq

from config import (
    allowedQuestionTypes,
    groqApiKey,
    groqModel,
    logger,
    maxInputCharacters,
    maxPdfPages,
)
from database import getDatabase, utcNowText

groqClient = Groq(api_key=groqApiKey)

promptsFolder = Path(__file__).parent / "prompts"


def readPrompt(fileName):
    return (promptsFolder / fileName).read_text(encoding="utf-8")


def logAiUsage(taskName, response, startTime):
    secondsTaken = time.time() - startTime
    usage = getattr(response, "usage", None)
    if usage:
        logger.info(
            f"AI call '{taskName}': {secondsTaken:.1f}s, "
            f"{usage.prompt_tokens} prompt tokens, {usage.completion_tokens} reply tokens"
        )

chunkSize = 8000
chunkOverlap = 500
maxChunks = 15
maxParallelGroqCalls = 4

questionGeneratorInstructions = readPrompt("questionGenerator.txt")
topicNameInstructions = readPrompt("topicName.txt").strip()


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
            startTime = time.time()
            response = groqClient.chat.completions.create(
                model=groqModel,
                messages=[
                    {"role": "system", "content": questionGeneratorInstructions},
                    {"role": "user", "content": f"Generate study questions from this material:\n\n{textChunk}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                extra_body={"reasoning_effort": "none"},
            )
            logAiUsage("generateQuestions", response, startTime)
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
        startTime = time.time()
        response = groqClient.chat.completions.create(
            model=groqModel,
            messages=[
                {"role": "system", "content": topicNameInstructions},
                {"role": "user", "content": textSample[:3000]},
            ],
            temperature=0.1,
            extra_body={"reasoning_effort": "none"},
        )
        logAiUsage("topicName", response, startTime)
        topicName = response.choices[0].message.content.strip().strip('"\'.')
        if topicName:
            return topicName[:100]
        return "Unknown Topic"
    except Exception as error:
        logger.warning(f"Failed to generate topic name: {error}")
        return "Unknown Topic"


def saveSourceAndQuestions(title, sourceType, characterCount, topicName, questions):
    now = utcNowText()
    with closing(getDatabase()) as connection:
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
