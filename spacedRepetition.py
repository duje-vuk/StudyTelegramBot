from datetime import timedelta

from database import utcNow

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
