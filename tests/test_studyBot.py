import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

import database
import questionGenerator
import spacedRepetition
import telegramBot


@pytest.fixture
def freshDatabase(tmp_path, monkeypatch):
    databaseFile = str(tmp_path / "test.db")
    monkeypatch.setattr(database, "databasePath", databaseFile)
    database.createTables()
    return databaseFile


def test_failedAnswerResetsProgress():
    easiness, interval, repetitions = spacedRepetition.calculateNextReview(1, 2.5, 10, 4)
    assert (interval, repetitions) == (1, 0)
    assert easiness >= 1.3


def test_correctAnswersGrowTheInterval():
    assert spacedRepetition.calculateNextReview(5, 2.5, 0, 0)[1:] == (1, 1)
    assert spacedRepetition.calculateNextReview(5, 2.5, 1, 1)[1:] == (3, 2)
    assert spacedRepetition.calculateNextReview(5, 2.5, 3, 2)[1] == 8


def test_intervalNeverPassesOneYear():
    assert spacedRepetition.calculateNextReview(5, 2.5, 300, 9)[1] == 365


def test_chunksOverlapAndStop():
    chunks = questionGenerator.splitIntoChunks("x" * 20000)
    assert len(chunks) == 3
    assert all(len(chunk) <= questionGenerator.chunkSize for chunk in chunks)
    assert questionGenerator.splitIntoChunks("short") == ["short"]


def test_cleanQuestionsDropsBadItems():
    reply = {"questions": [
        {"question": " What? ", "answer": " This ", "type": "factual"},
        {"question": "No answer"},
        {"question": " ", "answer": "x"},
        {"question": "Q", "answer": "A", "type": "weird"},
        "junk",
    ]}
    cleaned = questionGenerator.cleanQuestions(reply)
    assert cleaned == [
        {"question": "What?", "answer": "This", "type": "factual"},
        {"question": "Q", "answer": "A", "type": "unknown"},
    ]
    assert questionGenerator.cleanQuestions({"questions": "nope"}) == []


def test_oldDatabaseColumnsAreRenamed(tmp_path, monkeypatch):
    databaseFile = str(tmp_path / "old.db")
    connection = sqlite3.connect(databaseFile)
    connection.executescript("""
        CREATE TABLE sources (id INTEGER PRIMARY KEY, title TEXT NOT NULL, type TEXT NOT NULL DEFAULT 'text',
            char_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE questions (id INTEGER PRIMARY KEY, source_id INTEGER, question TEXT NOT NULL, answer TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'unknown', status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL,
            easiness_factor REAL NOT NULL DEFAULT 2.5, interval INTEGER NOT NULL DEFAULT 0,
            repetitions INTEGER NOT NULL DEFAULT 0, next_review TEXT, paused INTEGER NOT NULL DEFAULT 0);
        INSERT INTO sources VALUES (1, 'Notes', 'text', 10, '2026-01-01');
        INSERT INTO questions VALUES (1, 1, 'Q', 'A', 'factual', 'approved', '2026-01-01', 2.6, 3, 2, NULL, 0);
    """)
    connection.commit()
    connection.close()

    monkeypatch.setattr(database, "databasePath", databaseFile)
    database.createTables()

    connection = database.getDatabase()
    row = dict(connection.execute("SELECT * FROM questions").fetchone())
    connection.close()
    assert row["sourceId"] == 1
    assert row["easinessFactor"] == 2.6
    assert "source_id" not in row


def test_gradingUpdatesTheQuestion(freshDatabase):
    connection = database.getDatabase()
    connection.execute(
        "INSERT INTO questions (question, answer, status, createdAt) VALUES ('Q', 'A', 'approved', '2026-01-01')"
    )
    connection.commit()
    result = spacedRepetition.gradeQuestion(connection, 1, 5)
    connection.close()
    assert result["interval"] == 1
    assert result["repetitions"] == 1
    assert database.countDueQuestions() == 0


def test_botIgnoresStrangers(monkeypatch):
    monkeypatch.setattr(telegramBot, "telegramChatId", "12345")
    calls = []

    async def fakeHandler(update, context):
        calls.append(update.effective_chat.id)

    checkedHandler = telegramBot.onlyMyChat(fakeHandler)
    stranger = SimpleNamespace(effective_chat=SimpleNamespace(id=999))
    owner = SimpleNamespace(effective_chat=SimpleNamespace(id=12345))

    asyncio.run(checkedHandler(stranger, None))
    asyncio.run(checkedHandler(owner, None))
    assert calls == [12345]
