from datetime import datetime

from config import (
    logger,
    nudgeIntervalSeconds,
    studyHoursEnd,
    studyHoursStart,
    telegramBotToken,
    telegramChatId,
)
from database import (
    countDueQuestions,
    dueCondition,
    fetchDueQuestions,
    getDatabase,
    pauseAllQuestions,
    resumeAllQuestions,
    utcNowText,
)
from spacedRepetition import gradeQuestion

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler
    telegramInstalled = True
except ImportError:
    telegramInstalled = False

telegramEnabled = bool(
    telegramInstalled
    and telegramBotToken
    and telegramBotToken != "your_token_here"
    and telegramChatId
)

if telegramInstalled and not telegramEnabled:
    logger.warning("Telegram bot disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")


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
        chat_id=targetChatId or telegramChatId,
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

    if studySessions.get(telegramChatId):
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

    await context.bot.send_message(chat_id=telegramChatId, text="\n".join(lines))
    logger.info(f"Sent nudge: {total} due across {len(topicCounts)} topics.")


async def startTelegramBot():
    global telegramApp
    if not telegramEnabled:
        logger.info("Telegram bot not enabled. Skipping.")
        return

    logger.info("Starting embedded Telegram bot...")
    telegramApp = Application.builder().token(telegramBotToken).build()

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
