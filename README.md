# lcc-aldar-kose
"Aldark Kose Storyboard" project from HackNU 2025

## Docker Quickstart

```bash
docker-compose up --build -d
```

The command starts both services:

- Frontend: http://localhost:5173 (Vite dev server)
- Backend: http://localhost:8000 (FastAPI + SQLite)

The frontend runs with mocked assistant replies (`VITE_USE_MOCK=true`) so you always see responses while the real model is offline. Flip it to `false` in `docker-compose.yml` once the backend generates replies. The backend SQLite volume is persisted under `backend/data`. Stop the stack with `Ctrl+C`.
