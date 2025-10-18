# Aldar Kose Backend

FastAPI service that persists chat sessions, user prompts, and assistant replies (including image storyboards) in a SQLite database.

## Tech stack

- [FastAPI](https://fastapi.tiangolo.com/) for the REST API
- [SQLAlchemy 2.x](https://docs.sqlalchemy.org/) for ORM mapping
- SQLite (local file storage) by default; pluggable via `DATABASE_URL`

## Getting started

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API is served at `http://127.0.0.1:8000` by default. Visit `/docs` for interactive Swagger documentation.

## Configuration

- `DATABASE_URL` (optional): override the default `sqlite:///backend/data/app.db`. Any SQLAlchemy-supported URL works.
- CORS is open to all origins during development. Lock this down before production.

## Data model

```
Chat ─┬─< Message ─┬─< MessageAttachment
      │            └─ role: user | assistant
      └─ title          content: markdown/plain text
                        attachments: 0..* image URLs
```

## REST endpoints (v1)

- `POST   /api/v1/chats` – create a chat session (optional title)
- `GET    /api/v1/chats` – list chats with message counts & last activity
- `GET    /api/v1/chats/{chat_id}` – retrieve chat with its messages
- `DELETE /api/v1/chats/{chat_id}` – remove chat and cascading messages
- `GET    /api/v1/chats/{chat_id}/messages` – list messages in sequence order
- `POST   /api/v1/chats/{chat_id}/messages` – append user or assistant message, including 6–10 image attachments when needed

Messages accept payloads in the shape:

```json
{
  "role": "assistant",
  "content": "Visual storyboard inspired by ...",
  "attachments": [
    {
      "type": "image",
      "url": "https://picsum.photos/seed/example/512/384",
      "alt": "Frame 1"
    }
  ]
}
```

## Health check

`GET /health` returns `{ "status": "ok" }`.

## Development notes

- Tables auto-create on startup; delete `backend/data/app.db` to reset locally.
- Extend `app/crud.py` if you need streaming responses or analytics later.
- Consider adding Alembic migrations when the schema stabilises.
