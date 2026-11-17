import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone

from config import databasePath


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


def rowsToList(rows):
    return [dict(row) for row in rows]


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


questionWithSourceQuery = """
    SELECT questions.*, sources.title AS sourceTitle
    FROM questions
    LEFT JOIN sources ON questions.sourceId = sources.id
"""

dueCondition = "status = 'approved' AND paused = 0 AND (nextReview IS NULL OR nextReview <= ?)"


def fetchDueQuestions(connection, limit):
    rows = connection.execute(
        questionWithSourceQuery + f"""
        WHERE {dueCondition}
        ORDER BY questions.nextReview IS NULL DESC, questions.nextReview ASC
        LIMIT ?
        """,
        (utcNowText(), limit),
    ).fetchall()
    return rowsToList(rows)


def readStudyStats():
    oneWeekLater = (utcNow() + timedelta(days=7)).isoformat()

    with closing(getDatabase()) as connection:
        row = connection.execute("""
            SELECT
                SUM(CASE WHEN paused = 0 AND (nextReview IS NULL OR nextReview <= :now) THEN 1 ELSE 0 END) AS dueNow,
                COUNT(*) AS totalApproved,
                SUM(CASE WHEN repetitions > 0 THEN 1 ELSE 0 END) AS reviewed,
                SUM(CASE WHEN paused = 0 AND nextReview > :now AND nextReview <= :week THEN 1 ELSE 0 END) AS upcomingWeek,
                SUM(CASE WHEN paused = 1 THEN 1 ELSE 0 END) AS pausedCount
            FROM questions
            WHERE status = 'approved'
        """, {"now": utcNowText(), "week": oneWeekLater}).fetchone()

    return {
        "dueNow": row["dueNow"] or 0,
        "totalInRotation": row["totalApproved"] or 0,
        "reviewedAtLeastOnce": row["reviewed"] or 0,
        "upcomingSevenDays": row["upcomingWeek"] or 0,
        "paused": row["pausedCount"] or 0,
    }


def countDueQuestions():
    with closing(getDatabase()) as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM questions WHERE {dueCondition}", (utcNowText(),)).fetchone()
        return row[0]


def pauseAllQuestions():
    with closing(getDatabase()) as connection:
        result = connection.execute(
            "UPDATE questions SET paused = 1, nextReview = NULL WHERE status = 'approved' AND paused = 0"
        )
        connection.commit()
        return result.rowcount


def resumeAllQuestions():
    with closing(getDatabase()) as connection:
        result = connection.execute(
            "UPDATE questions SET paused = 0, nextReview = ? WHERE status = 'approved' AND paused = 1",
            (utcNowText(),),
        )
        connection.commit()
        return result.rowcount
