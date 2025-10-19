# lcc-aldar-kose
"Aldark Kose Storyboard" project from HackNU 2025

![](aldar.png)

![alt text](image.png)

### Colab

[📓 Notebook without integrated prompt optimizations](txt2img_img2img_inference.ipynb)

## Docker Quickstart

```bash
docker compose up --build -d
```

The command starts both services:

- Frontend: http://localhost:5173 (Vite dev server)
- Backend: http://localhost:8000 (FastAPI + SQLite)

The frontend runs with mocked assistant replies (`VITE_USE_MOCK=true`) so you always see responses while the real model is offline. Flip it to `false` in `docker-compose.yml` once the backend generates replies. The backend SQLite volume is persisted under `backend/data`. Stop the stack with `Ctrl+C`.

## ML Microservice

The GPU inference service lives under `ml/service.py`. It handles storyboard generation, talks to the backend for status updates, and streams image URLs from MinIO/S3. To run it locally:

```bash
pip install -r ml/requirements.txt
python ml/service.py
```

The service expects MinIO (or an S3-compatible store) to be available. Populate a `.env` file based on `.env.example`, making sure these variables point at your MinIO instance and OpenAI creds:

- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_ENDPOINT_URL`
- `OPENAI_API_KEY`

Start MinIO before launching the service (default endpoint in the example is `http://localhost:9000`). Once the microservice is up, the backend can dispatch generation jobs to it automatically.
