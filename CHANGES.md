# StudyBot Performance Refactor — Changes

Date: 2026-07-07. All changes in `app.py` (backend + embedded frontend). Behavior preserved; speed and responsiveness improved.

## Backend — Speed

### 1. Parallel Groq chunk processing (biggest win)
**Before:** Chunks sent to Groq one at a time with a forced 2s sleep between each (`DELAY_BETWEEN_CALLS`). A 15-chunk document = 15 sequential API calls + 28s of pure sleep.
**After:** All chunks run concurrently through `asyncio.gather` bounded by a semaphore (`GROQ_CONCURRENCY = 4`). Fixed sleeps removed; 429 rate limits still handled by existing exponential backoff. Topic-name generation also runs concurrently with chunk calls instead of before them.
**Est. effect:** large documents generate ~4-8x faster.

### 2. PDF parsing off the event loop
**Before:** `extract_pdf_text()` (CPU-bound pymupdf) ran directly inside the async endpoint, freezing the whole server during a 20MB parse.
**After:** `await asyncio.to_thread(extract_pdf_text, ...)`. Server stays responsive during uploads.

### 3. PDF upload streamed with hard cap
**Before:** `await file.read()` loaded the entire upload into memory before checking size.
**After:** Read in 1MB pieces; abort with 413 as soon as the cap is crossed.

### 4. Blocking DB endpoints moved off the event loop
**Before:** All endpoints were `async def` but did synchronous sqlite I/O, blocking the event loop.
**After:** Pure-DB endpoints (`/api/questions` GET, `/api/questions/{id}` DELETE, `/api/sources`, `/api/due`, `/api/stats`, `/api/upcoming`, `/api/pause`, `/api/resume`, `/api/counts`) are now sync `def` — FastAPI automatically runs them in its threadpool.

### 5. `/api/stats` collapsed from 5 queries to 1
Single `SUM(CASE ...)` scan replaces five separate `COUNT(*)` queries per call.

### 6. New `/api/counts` endpoint
Returns `{total, pending, approved, due}` via one COUNT query. Replaces the frontend pattern of downloading the entire question table (with JOINs) just to count rows.

### 7. Batch insert of generated questions
`executemany` instead of a per-question `execute` loop.

### 8. SQLite tuning
- New composite index `idx_questions_due (status, paused, next_review)` matching the due-question WHERE clause exactly.
- `PRAGMA journal_mode=WAL` moved to `init_db()` (it's persistent; was being re-run on every connection).
- `PRAGMA synchronous=NORMAL` added (safe with WAL, faster writes).
- Connection `timeout=10` to avoid "database is locked" under concurrent web + Telegram writes.

## Backend — Bug fixes

### 9. `chunk_text` trailing-chunk bug
**Before:** If text length landed within the overlap window, a tiny final chunk consisting only of already-processed overlap text was emitted → one wasted Groq API call per document.
**After:** Loop breaks when the chunk reaches end of text.

### 10. `/api/due?limit=` negative-value hole
**Before:** `min(limit, 100)` — a negative limit becomes `LIMIT -n`, which sqlite treats as *no limit* (full table dump).
**After:** `max(1, min(limit, 100))`.

### 11. Telegram session recursion
**Before:** `_send_next_in_session` called itself recursively for every deleted question in the queue — deep recursion on stale queues; also reopened the DB per skip.
**After:** Iterative loop, one DB connection; correctly ends the session if only deleted questions remain.

### 12. Deprecated APIs replaced
- `@app.on_event("startup"/"shutdown")` → FastAPI `lifespan` context manager.
- `asyncio.get_event_loop()` / `run_in_executor` → `asyncio.to_thread`.
- Frontend: implicit global `event` in `setBankFilter` → explicit `this` parameter.

### 13. Logging cleanup
`print()` calls in the Groq path replaced with `logger` (raw responses now `debug`-level instead of dumping 500 chars to stdout per chunk); `import time` moved from inside `call_groq` to module top.

### 14. `/api/upcoming` bounded and slimmed
Added `LIMIT 500` — previously unbounded, could return the entire future schedule. Also dropped `answer`, `easiness_factor`, `repetitions` from the SELECT: the Upcoming view never rendered them, and full answer text alone could be megabytes of dead payload.

### 14a. Shared JSON body parsing
**Before:** Four endpoints duplicated body parsing with `except (json.JSONDecodeError, Exception)` — the second clause swallowed *everything*, mislabeling unrelated failures as "Invalid JSON". A non-object body like `[1,2,3]` passed parsing and crashed `.get()` with a 500.
**After:** One `read_json_body()` helper: size cap, precise `JSONDecodeError` handling, and explicit rejection of non-object JSON with a 400.

### 14b. DST bug in nudge scheduler
**Before:** `_scheduled_check` hardcoded `CET = UTC+1`, so the 8:00–23:00 study-hours window drifted 1 hour every summer.
**After:** Uses the system local timezone (`datetime.now().astimezone()`).

## Frontend — Responsiveness

### 15. Badge updates
**Before:** `updateBadges()` fetched *every question in the DB* plus the due list (2 heavy requests) just to display 3 numbers. Ran after every action.
**After:** One call to `/api/counts`.

### 16. Question Bank load
**Before:** When filtered, fetched the full question table a *second* time just for stat boxes; requests sequential.
**After:** Question list + `/api/counts` fetched in parallel (`Promise.all`); no second full-table download.

### 17. Study view load
**Before:** `/api/stats` → then `/api/due` → then `/api/upcoming`, strictly sequential.
**After:** Stats + due fetched in parallel; upcoming loads independently.

### 17a. Due badge showed wrong count
**Before:** Badge displayed `(/api/due).length`, which the API caps at 20 — "20 due" shown even with 200 questions due.
**After:** Badge uses the true `due_now` from `/api/stats` (and `due` from `/api/counts`).

### 18. Instant grading
**Before:** After grading a flashcard the UI waited for the network round-trip before showing the next card.
**After:** Optimistic UI — next card appears immediately; grade posts in the background.

### 19. In-place card removal
**Before:** Approving or deleting a single question re-fetched and re-rendered the entire list.
**After:** The single card is removed from the DOM; selection state updated. (Bank-view status changes still reload, since the tag text changes.)

### 20. Batched DOM rendering
Review list, bank list, and upcoming list now build into a `DocumentFragment` and insert once — one reflow instead of one per card.

## Not changed (noted for awareness)
- SM-2 math and Telegram requeue behavior — work as designed ("Easy" cards intentionally reappear once per session).
- DB schema — only an additional index; existing `spaced_rep.db` files keep working.
- No auth on API endpoints — fine for localhost use, don't expose publicly.
- `quickstart.html`, `study-bot.service`, `requirements.txt` untouched (no issues found).

## Verification
- `python -m py_compile` — passes.
- Full AST parse — passes (47 function defs).
- Embedded frontend JS extracted and checked with `node --check` — passes.
- Grep confirms no stale references (`DELAY_BETWEEN_CALLS`, `on_event`, `get_event_loop` all gone).
- Unit checks (stdlib-only):
  - `chunk_text`: empty input, exact-boundary input (no duplicate tail), overlap correctness, `MAX_CHUNKS` cap;
  - `sm2`: interval progression 1 → 3 → EF-scaled, fail-reset, EF floor 1.3, 365-day cap for all quality values;
  - `validate_questions`: malformed entries dropped, unknown types normalized.
- All new/changed SQL executed against a seeded in-memory replica of the schema — `/api/stats`, `/api/counts`, `/api/upcoming` return correct values; `EXPLAIN QUERY PLAN` confirms the due-question query uses the new `idx_questions_due` composite index.
- Live-server boot not possible in the review sandbox (no package installs); recommend one local smoke run: `uvicorn app:app` → generate, review, study.
