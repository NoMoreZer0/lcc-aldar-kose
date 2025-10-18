# Aldar Kose Storyboard Generator - Implementation Plan (GPU Microservice Architecture)

**Project:** AI-powered storyboard generation from user prompts
**Target:** Hackathon demo (<1 day implementation)
**Stack:** FastAPI Backend + GPU Microservice + React Frontend
**Architecture:** Microservice pattern (production-ready, scalable)

---

## Architecture Overview

```
┌─────────────────┐
│  React Frontend │  http://localhost:5173
│  (TypeScript)   │  - Submit prompts
└────────┬────────┘  - Poll job status
         │           - Display carousel
         │ HTTP REST
         ▼
┌─────────────────┐
│ FastAPI Backend │  http://localhost:8000
│  + SQLite DB    │  - Job management
└────────┬────────┘  - Status tracking
         │           - Image URLs
         │ HTTP async
         ▼
┌─────────────────┐
│  GPU Service    │  http://gpu-server:8001
│  (Microservice) │  - Image generation
└─────────────────┘  - Stateless inference
```

**Flow:**
1. User submits prompt via frontend
2. Backend creates job record, calls GPU service (async HTTP)
3. GPU service generates images, returns URLs
4. Backend updates job status
5. Frontend polls job status, shows progress
6. On completion, frontend displays images in cinematic carousel

**Why Microservice Pattern:**
- ✅ Production-ready architecture (impresses AI company judges)
- ✅ GPU can be on separate server/cluster
- ✅ Scalable (add more GPU workers behind load balancer)
- ✅ Stateless inference service (easier to debug/test)
- ✅ Failure isolation (GPU crash doesn't kill backend)
- ✅ Simpler than RabbitMQ (3-4 hours vs 6+ hours)

---

## TASK 1: Backend - Job Management System

### Context
**Current State:**
- Backend at `/backend` uses FastAPI with SQLAlchemy 2.x
- Existing models: `Chat`, `Message`, `MessageAttachment` in `/backend/app/models.py`
- Database: SQLite at `/backend/data/app.db`
- Existing schemas in `/backend/app/schemas.py` use Pydantic with `orm_mode = True`
- Main app in `/backend/app/main.py` with CORS enabled
- API router structure: `/backend/app/api/v1/chats.py`

### Objective
Add job management system to track storyboard generation requests and their status.

### Technical Specifications

#### 1.1 Database Model (`/backend/app/models.py`)

**Add to existing models:**

```python
from sqlalchemy import JSON

class JobStatus(str, PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    chat_id: Mapped[str] = mapped_column(String(36), ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("messages.id", ondelete="SET NULL"), nullable=True, index=True)

    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus, name="job_status"), default=JobStatus.PENDING)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0-100

    # Store result image URLs as JSON array
    result_urls: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Generation parameters
    num_frames: Mapped[int] = mapped_column(Integer, default=8)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    chat: Mapped[Chat] = relationship(back_populates="jobs")

# Add to Chat model:
class Chat(Base):
    # ... existing fields ...
    jobs: Mapped[List["Job"]] = relationship(back_populates="chat", cascade="all, delete-orphan")
```

**Why these fields:**
- `status`: Tracks generation lifecycle (pending → processing → completed/failed)
- `progress`: For real-time UI updates (0-100%)
- `result_urls`: JSON array of image URLs when completed
- `num_frames`: Configurable 6-10 frames

#### 1.2 Pydantic Schemas (`/backend/app/schemas.py`)

**Add schemas:**

```python
from typing import List, Literal, Optional
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field

JobStatusType = Literal["pending", "processing", "completed", "failed"]

class JobCreate(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Storyboard generation prompt")
    num_frames: int = Field(default=8, ge=6, le=10, description="Number of frames to generate")

class JobUpdate(BaseModel):
    status: Optional[JobStatusType] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    result_urls: Optional[List[str]] = None
    error_message: Optional[str] = None

class JobRead(BaseModel):
    id: UUID
    chat_id: UUID
    prompt: str
    status: JobStatusType
    progress: int
    result_urls: Optional[List[str]] = None
    error_message: Optional[str] = None
    num_frames: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        orm_mode = True
```

#### 1.3 CRUD Operations (`/backend/app/crud.py`)

**Add to existing crud.py:**

```python
from typing import Optional
from .models import Job, JobStatus

class JobNotFoundError(Exception):
    """Raised when job is not found."""
    pass

def create_job(db: Session, chat_id: str, job_in: schemas.JobCreate) -> Job:
    """Create a new storyboard generation job."""
    # Verify chat exists
    chat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not chat:
        raise ChatNotFoundError(f"Chat {chat_id} not found")

    job = Job(
        chat_id=chat_id,
        prompt=job_in.prompt,
        num_frames=job_in.num_frames,
        status=JobStatus.PENDING,
        progress=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job

def get_job(db: Session, job_id: str) -> Job:
    """Get job by ID."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise JobNotFoundError(f"Job {job_id} not found")
    return job

def update_job(db: Session, job_id: str, job_update: schemas.JobUpdate) -> Job:
    """Update job status and progress."""
    job = get_job(db, job_id)

    update_data = job_update.dict(exclude_unset=True)

    # Set completed_at when status changes to completed/failed
    if "status" in update_data and update_data["status"] in ["completed", "failed"]:
        from datetime import datetime, timezone
        job.completed_at = datetime.now(timezone.utc)

    for field, value in update_data.items():
        setattr(job, field, value)

    db.commit()
    db.refresh(job)
    return job

def list_jobs_by_chat(db: Session, chat_id: str) -> list[Job]:
    """List all jobs for a chat, newest first."""
    return db.query(Job).filter(Job.chat_id == chat_id).order_by(Job.created_at.desc()).all()
```

#### 1.4 GPU Service Client (`/backend/app/gpu_client.py` - NEW FILE)

```python
"""Client for GPU inference microservice."""

import os
import httpx
import logging
from typing import Optional

logger = logging.getLogger(__name__)

GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://localhost:8001")
GPU_SERVICE_TIMEOUT = float(os.getenv("GPU_SERVICE_TIMEOUT", "300.0"))  # 5 minutes


async def submit_generation_job(job_id: str, prompt: str, num_frames: int) -> bool:
    """
    Submit generation job to GPU microservice.

    Returns True if successfully submitted, False otherwise.
    GPU service will process async and callback to update job status.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVICE_URL}/generate",
                json={
                    "job_id": job_id,
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "callback_url": os.getenv("BACKEND_CALLBACK_URL", "http://backend:8000/api/v1")
                }
            )

            response.raise_for_status()
            logger.info(f"Submitted job {job_id} to GPU service")
            return True

    except httpx.TimeoutException:
        logger.error(f"GPU service timeout for job {job_id}")
        return False
    except httpx.HTTPError as e:
        logger.error(f"GPU service HTTP error for job {job_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error submitting job {job_id}: {e}")
        return False


async def check_gpu_service_health() -> bool:
    """Check if GPU service is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{GPU_SERVICE_URL}/health")
            return response.status_code == 200
    except Exception:
        return False
```

#### 1.5 API Endpoints (`/backend/app/api/v1/jobs.py` - NEW FILE)

```python
"""Job management API endpoints."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.orm import Session

from ... import crud, schemas
from ...database import get_db
from ...gpu_client import submit_generation_job
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post(
    "/chats/{chat_id}/generate",
    response_model=schemas.JobRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create storyboard generation job"
)
async def create_generation_job(
    job_in: schemas.JobCreate,
    chat_id: UUID = Path(..., description="Chat ID to associate job with"),
    db: Session = Depends(get_db),
) -> schemas.JobRead:
    """
    Submit a new storyboard generation job.

    Creates a job record and submits to GPU microservice for async processing.
    Returns immediately with job_id for status polling.
    """
    try:
        # Create job in database
        job = crud.create_job(db, str(chat_id), job_in)

        # Submit to GPU service asynchronously
        submitted = await submit_generation_job(
            job_id=str(job.id),
            prompt=job.prompt,
            num_frames=job.num_frames
        )

        if not submitted:
            logger.warning(f"Job {job.id} created but GPU service submission failed")
            # Update job status to failed
            crud.update_job(db, str(job.id), schemas.JobUpdate(
                status="failed",
                error_message="GPU service unavailable. Please try again."
            ))

        # Return current job state
        return schemas.JobRead.from_orm(crud.get_job(db, str(job.id)))

    except crud.ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Error creating job: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create job: {str(exc)}"
        ) from exc


@router.get(
    "/{job_id}",
    response_model=schemas.JobRead,
    summary="Get job status"
)
def get_job_status(
    job_id: UUID = Path(..., description="Job ID"),
    db: Session = Depends(get_db),
) -> schemas.JobRead:
    """
    Poll job status and progress.

    Frontend should poll this endpoint every 2-3 seconds while job is processing.
    """
    try:
        job = crud.get_job(db, str(job_id))
        return schemas.JobRead.from_orm(job)
    except crud.JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch(
    "/{job_id}",
    response_model=schemas.JobRead,
    summary="Update job (callback from GPU service)"
)
def update_job_status(
    job_update: schemas.JobUpdate,
    job_id: UUID = Path(..., description="Job ID"),
    db: Session = Depends(get_db),
) -> schemas.JobRead:
    """
    Update job status and progress.

    This endpoint is called by the GPU microservice to report progress/completion.
    Can also be called by frontend for testing/admin purposes.
    """
    try:
        job = crud.update_job(db, str(job_id), job_update)
        logger.info(f"Job {job_id} updated: status={job.status}, progress={job.progress}")
        return schemas.JobRead.from_orm(job)
    except crud.JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/chats/{chat_id}/jobs",
    response_model=List[schemas.JobRead],
    summary="List chat jobs"
)
def list_chat_jobs(
    chat_id: UUID = Path(..., description="Chat ID"),
    db: Session = Depends(get_db),
) -> List[schemas.JobRead]:
    """Get all generation jobs for a chat."""
    jobs = crud.list_jobs_by_chat(db, str(chat_id))
    return [schemas.JobRead.from_orm(job) for job in jobs]
```

#### 1.6 Router Registration (`/backend/app/api/router.py`)

**Modify existing router.py:**

```python
from fastapi import APIRouter
from .v1 import chats
from .v1 import jobs  # ADD THIS

api_router = APIRouter()
api_router.include_router(chats.router)
api_router.include_router(jobs.router)  # ADD THIS
```

#### 1.7 Static File Serving (`/backend/app/main.py`)

**Add to main.py after app creation:**

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

# ... existing app creation ...

# Serve generated images as static files
# Images will be at /images/{run_id}/frame_XX.png
images_dir = Path(os.getenv("IMAGES_DIR", "/app/images"))
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

# Add health endpoint
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

#### 1.8 Dependencies (`/backend/requirements.txt`)

**Add:**
```
httpx==0.25.2
```

### Acceptance Criteria
- [ ] Job model exists in database with all fields
- [ ] API endpoints: POST /jobs/chats/{id}/generate, GET /jobs/{id}, PATCH /jobs/{id}
- [ ] GPU client can submit jobs via HTTP
- [ ] Static images served at /images/ path
- [ ] Health check endpoint works

### Testing
```bash
# Start backend
cd backend
uvicorn app.main:app --reload

# Create job
curl -X POST http://localhost:8000/api/v1/jobs/chats/{chat_id}/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Aldar Kose tricks a merchant", "num_frames": 8}'

# Check status
curl http://localhost:8000/api/v1/jobs/{job_id}
```

---

## TASK 2: GPU Inference Microservice

### Context
**Current State:**
- ML code at `/ml/src/` with existing storyboard generation pipeline
- CLI tool: `/ml/src/cli/generate_storyboard.py` generates images from prompts
- Outputs saved to `/ml/outputs/{run_id}/frame_XX.png`
- Pipeline uses: `StoryboardGenerationPipeline`, `plan_shots` LLM planner
- Config at `/ml/configs/default.yaml`
- Requirements in `/ml/requirements.txt`

### Objective
Create lightweight FastAPI microservice that wraps the ML pipeline and exposes REST API for storyboard generation. Runs on GPU server independently from main backend.

### Technical Specifications

#### 2.1 Microservice App (`/ml/service.py` - NEW FILE)

```python
"""
GPU Inference Microservice for Storyboard Generation.

Stateless FastAPI service that accepts generation requests,
runs ML pipeline, uploads results, and callbacks to main backend.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import time

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.diffusion.pipeline import StoryboardGenerationPipeline
from src.llm.planner import plan_shots
from src.utils import io as io_utils
from src.utils.determinism import configure_determinism, set_seed
from src.utils.schema import ConfigModel

# Configuration
CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"
OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://localhost:8000/images")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aldar Kose GPU Inference Service", version="1.0.0")


class GenerationRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier from backend")
    prompt: str = Field(..., min_length=1, max_length=2000)
    num_frames: int = Field(default=8, ge=6, le=10)
    callback_url: str = Field(..., description="Backend URL to report completion")


class GenerationResponse(BaseModel):
    status: str
    job_id: str
    message: str


async def report_progress(callback_url: str, job_id: str, progress: int, status: str = "processing"):
    """Report progress to backend."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{callback_url}/jobs/{job_id}",
                json={"status": status, "progress": progress}
            )
            logger.info(f"Reported progress for {job_id}: {progress}%")
    except Exception as e:
        logger.error(f"Failed to report progress for {job_id}: {e}")


async def report_completion(callback_url: str, job_id: str, image_urls: list):
    """Report successful completion to backend."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{callback_url}/jobs/{job_id}",
                json={
                    "status": "completed",
                    "progress": 100,
                    "result_urls": image_urls
                }
            )
            logger.info(f"Reported completion for {job_id}")
    except Exception as e:
        logger.error(f"Failed to report completion for {job_id}: {e}")


async def report_failure(callback_url: str, job_id: str, error_message: str):
    """Report failure to backend."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{callback_url}/jobs/{job_id}",
                json={
                    "status": "failed",
                    "progress": 0,
                    "error_message": error_message
                }
            )
            logger.error(f"Reported failure for {job_id}: {error_message}")
    except Exception as e:
        logger.error(f"Failed to report failure for {job_id}: {e}")


async def run_generation(req: GenerationRequest):
    """
    Run storyboard generation pipeline.

    This runs in background task, updates backend via callbacks.
    """
    job_id = req.job_id
    prompt = req.prompt
    num_frames = req.num_frames
    callback_url = req.callback_url

    logger.info(f"Starting generation for job {job_id}: {num_frames} frames for '{prompt}'")

    try:
        # Load config
        config = ConfigModel.from_path(CONFIG_PATH)
        config_dict = config.model_dump() if hasattr(config, "model_dump") else config.dict()

        configure_determinism(True)
        base_seed = set_seed(None)

        # Progress: 10% - Planning shots
        await report_progress(callback_url, job_id, 10)

        logger.info(f"Planning {num_frames} shots for job {job_id}...")
        shots = plan_shots(prompt, n_frames=num_frames)

        # Progress: 30% - Shots planned, starting generation
        await report_progress(callback_url, job_id, 30)

        # Setup output directory
        run_id = f"job_{job_id[:8]}_{int(time.time())}"
        output_dir = OUTPUT_BASE / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating {num_frames} frames for job {job_id}...")

        # Initialize and run pipeline
        pipeline = StoryboardGenerationPipeline(config_dict, logger=logger)

        index_payload = pipeline.run(
            logline=prompt,
            shots=shots,
            output_dir=output_dir,
            use_ip_adapter=config.model.get("use_ip_adapter", False),
            use_controlnet=config.model.get("use_controlnet", False),
            base_seed=base_seed,
        )

        # Progress: 90% - Generation complete, finalizing
        await report_progress(callback_url, job_id, 90)

        # Save index.json
        index_path = output_dir / "index.json"
        index_dict = index_payload.model_dump() if hasattr(index_payload, "model_dump") else index_payload.dict()
        io_utils.dump_json(index_path, index_dict)

        # Collect image URLs
        image_urls = []
        for frame_file in sorted(output_dir.glob("frame_*.png")):
            # URL: /images/{run_id}/frame_XX.png
            url_path = f"/images/{run_id}/{frame_file.name}"
            image_urls.append(url_path)

        logger.info(f"Job {job_id} completed: {len(image_urls)} frames generated")

        # Report completion
        await report_completion(callback_url, job_id, image_urls)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await report_failure(callback_url, job_id, str(e))


@app.post("/generate", response_model=GenerationResponse, status_code=202)
async def generate_storyboard(
    request: GenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit storyboard generation request.

    Returns immediately with 202 Accepted.
    Generation runs in background and callbacks to provided URL.
    """
    logger.info(f"Received generation request for job {request.job_id}")

    # Add to background tasks
    background_tasks.add_task(run_generation, request)

    return GenerationResponse(
        status="accepted",
        job_id=request.job_id,
        message=f"Generation started for {request.num_frames} frames"
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gpu-inference",
        "gpu_available": check_gpu_available()
    }


def check_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Aldar Kose GPU Inference Microservice",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        log_level="info"
    )
```

#### 2.2 GPU Service Dockerfile (`/ml/Dockerfile.service` - NEW FILE)

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir fastapi uvicorn httpx

# Copy ML source code
COPY . .

# Create outputs directory
RUN mkdir -p /app/outputs

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run service
CMD ["python3", "service.py"]
```

#### 2.3 GPU Service Requirements (`/ml/requirements.txt`)

**Add to existing requirements:**
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
```

#### 2.4 Standalone Deployment Script (`/ml/deploy_gpu_service.sh` - NEW FILE)

```bash
#!/bin/bash
# Deploy GPU service on remote GPU server

set -e

echo "🚀 Deploying Aldar Kose GPU Inference Service..."

# Configuration
PORT=${PORT:-8001}
WORKERS=${WORKERS:-1}

# Build Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.service -t aldar-kose-gpu:latest .

# Stop existing container if running
echo "🛑 Stopping existing container..."
docker stop aldar-kose-gpu 2>/dev/null || true
docker rm aldar-kose-gpu 2>/dev/null || true

# Run container with GPU support
echo "▶️  Starting GPU service container..."
docker run -d \
  --name aldar-kose-gpu \
  --gpus all \
  -p ${PORT}:8001 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/configs:/app/configs:ro \
  -e OUTPUT_DIR=/app/outputs \
  -e IMAGE_BASE_URL=${IMAGE_BASE_URL:-http://localhost:8000/images} \
  --restart unless-stopped \
  aldar-kose-gpu:latest

echo "✅ GPU service deployed successfully!"
echo "📊 Service running at http://localhost:${PORT}"
echo "🏥 Health check: http://localhost:${PORT}/health"
echo ""
echo "📋 View logs: docker logs -f aldar-kose-gpu"
echo "🛑 Stop service: docker stop aldar-kose-gpu"
```

Make executable:
```bash
chmod +x /ml/deploy_gpu_service.sh
```

### Acceptance Criteria
- [ ] GPU service runs on port 8001
- [ ] POST /generate accepts jobs and returns 202 Accepted
- [ ] Background task generates images and saves to /outputs
- [ ] Service callbacks to backend with progress updates
- [ ] Health check endpoint reports GPU availability
- [ ] Docker container has GPU access

### Testing

**Local testing (without GPU):**
```bash
cd ml
python3 service.py
```

**With Docker + GPU:**
```bash
cd ml
./deploy_gpu_service.sh
```

**Test generation:**
```bash
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-123",
    "prompt": "Test storyboard",
    "num_frames": 6,
    "callback_url": "http://backend:8000/api/v1"
  }'

# Check health
curl http://localhost:8001/health
```

---

## TASK 3: Frontend - Job Submission & Polling

### Context
**Current State:**
- React + TypeScript + Vite frontend at `/frontend`
- Custom hook: `/frontend/src/hooks/useChat.ts` manages chat state
- API calls use fetch with base URL from `VITE_API_BASE` env var
- Types defined in `/frontend/src/types.ts`
- Mock responses in `/frontend/src/mockResponses.ts` (currently enabled)

### Objective
Add storyboard generation job submission and real-time status polling to the chat interface.

### Technical Specifications

#### 3.1 TypeScript Types (`/frontend/src/types.ts`)

**Add to existing types:**

```typescript
export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface StoryboardJob {
  id: string;
  chatId: string;
  prompt: string;
  status: JobStatus;
  progress: number;
  resultUrls: string[] | null;
  errorMessage: string | null;
  numFrames: number;
  createdAt: string;
  updatedAt: string;
  completedAt: string | null;
}

export interface JobCreateRequest {
  prompt: string;
  numFrames?: number;
}
```

#### 3.2 Job Polling Hook (`/frontend/src/hooks/useJobPolling.ts` - NEW FILE)

```typescript
import { useCallback, useEffect, useRef, useState } from 'react';
import type { StoryboardJob } from '../types';

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api/v1').replace(/\/+$/, '');
const POLL_INTERVAL = 2000; // 2 seconds
const TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

interface UseJobPollingOptions {
  jobId: string | null;
  onComplete?: (job: StoryboardJob) => void;
  onError?: (error: string) => void;
}

export const useJobPolling = ({ jobId, onComplete, onError }: UseJobPollingOptions) => {
  const [job, setJob] = useState<StoryboardJob | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const intervalRef = useRef<number | null>(null);
  const timeoutRef = useRef<number | null>(null);

  const fetchJob = useCallback(async (id: string): Promise<StoryboardJob | null> => {
    try {
      const response = await fetch(`${API_BASE}/jobs/${id}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch job: ${response.status}`);
      }

      const data = await response.json();

      // Convert snake_case to camelCase
      return {
        id: data.id,
        chatId: data.chat_id,
        prompt: data.prompt,
        status: data.status,
        progress: data.progress,
        resultUrls: data.result_urls,
        errorMessage: data.error_message,
        numFrames: data.num_frames,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        completedAt: data.completed_at,
      };
    } catch (error) {
      console.error('Error fetching job:', error);
      return null;
    }
  }, []);

  const poll = useCallback(async () => {
    if (!jobId) return;

    const jobData = await fetchJob(jobId);
    if (!jobData) {
      onError?.('Failed to fetch job status');
      return;
    }

    setJob(jobData);

    // Stop polling if job is complete or failed
    if (jobData.status === 'completed' || jobData.status === 'failed') {
      setIsPolling(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      if (jobData.status === 'completed') {
        onComplete?.(jobData);
      } else {
        onError?.(jobData.errorMessage || 'Job failed');
      }
    }
  }, [jobId, fetchJob, onComplete, onError]);

  // Start polling when jobId changes
  useEffect(() => {
    if (!jobId) {
      setIsPolling(false);
      setJob(null);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      return;
    }

    setIsPolling(true);

    // Immediate first poll
    poll();

    // Set up polling interval
    intervalRef.current = window.setInterval(poll, POLL_INTERVAL);

    // Set up timeout
    timeoutRef.current = window.setTimeout(() => {
      setIsPolling(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      onError?.('Generation timed out after 10 minutes. Please try again.');
    }, TIMEOUT_MS);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [jobId, poll, onError]);

  return {
    job,
    isPolling,
  };
};
```

#### 3.3 Enhanced useChat Hook (`/frontend/src/hooks/useChat.ts`)

**Add to existing useChat hook:**

```typescript
// At the top, add import
import type { JobCreateRequest, StoryboardJob } from '../types';

// Inside UseChatResult interface, add:
interface UseChatResult {
  // ... existing fields ...
  generateStoryboard: (prompt: string, numFrames?: number) => Promise<StoryboardJob>;
}

// Inside useChat function, add new method:
const generateStoryboard = useCallback(
  async (prompt: string, numFrames: number = 8): Promise<StoryboardJob> => {
    setError(null);

    try {
      const targetChatId = await ensureChatId();

      const payload: JobCreateRequest = {
        prompt: prompt.trim(),
        numFrames,
      };

      const response = await fetch(buildUrl(`/jobs/chats/${targetChatId}/generate`), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`Failed to create generation job: ${response.status}`);
      }

      const data = await response.json();

      // Convert to camelCase
      const job: StoryboardJob = {
        id: data.id,
        chatId: data.chat_id,
        prompt: data.prompt,
        status: data.status,
        progress: data.progress,
        resultUrls: data.result_urls,
        errorMessage: data.error_message,
        numFrames: data.num_frames,
        createdAt: data.created_at,
        updatedAt: data.updated_at,
        completedAt: data.completed_at,
      };

      return job;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to generate storyboard. Please try again.';
      setError(message);
      throw err instanceof Error ? err : new Error(message);
    }
  },
  [ensureChatId]
);

// Add to return statement:
return {
  // ... existing fields ...
  generateStoryboard,
};
```

### Acceptance Criteria
- [ ] `useJobPolling` hook polls job status every 2 seconds
- [ ] Polling stops when job status is 'completed' or 'failed'
- [ ] Timeout after 10 minutes with helpful error message
- [ ] `generateStoryboard` method creates job and returns job object
- [ ] Types properly convert snake_case API to camelCase frontend

### Testing
```typescript
// In App.tsx or test component:
const { generateStoryboard } = useChat();

const handleGenerate = async () => {
  const job = await generateStoryboard("Test prompt", 8);
  console.log("Job created:", job.id);
};
```

---

## TASK 4: Frontend - Generation Progress UI

### Context
**Current State:**
- Message components in `/frontend/src/components/MessageBubble.tsx`
- Loading indicator exists: `.message-bubble--loading` with animated dots
- Styles in `/frontend/src/styles.css`
- Messages displayed in `/frontend/src/components/MessageList.tsx`

### Objective
Create real-time progress UI component that shows generation status with animated progress bar and status text.

### Technical Specifications

#### 4.1 Progress Component (`/frontend/src/components/GenerationProgress.tsx` - NEW FILE)

```typescript
import type { StoryboardJob } from '../types';

interface GenerationProgressProps {
  job: StoryboardJob;
}

const getStatusText = (status: StoryboardJob['status'], progress: number): string => {
  if (status === 'pending') {
    return 'Queued for processing...';
  }

  if (status === 'processing') {
    if (progress < 15) {
      return 'Initializing GPU pipeline...';
    } else if (progress < 35) {
      return 'AI planning narrative shots...';
    } else if (progress < 90) {
      const estimatedFrame = Math.floor((progress - 30) / 60 * 8);
      return `Generating frame ${estimatedFrame}/8...`;
    } else {
      return 'Finalizing storyboard...';
    }
  }

  if (status === 'completed') {
    return 'Storyboard ready!';
  }

  return 'Processing...';
};

export const GenerationProgress = ({ job }: GenerationProgressProps) => {
  const statusText = getStatusText(job.status, job.progress);
  const isActive = job.status === 'pending' || job.status === 'processing';

  return (
    <div className="generation-progress">
      <div className="generation-progress__header">
        <div className="generation-progress__icon">
          <svg
            className={isActive ? 'generation-progress__icon--spinning' : ''}
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <rect x="2" y="6" width="4" height="12" rx="1" fill="currentColor" opacity="0.8" />
            <rect x="7" y="4" width="4" height="16" rx="1" fill="currentColor" opacity="0.6" />
            <rect x="12" y="6" width="4" height="12" rx="1" fill="currentColor" opacity="0.8" />
            <rect x="17" y="8" width="4" height="8" rx="1" fill="currentColor" opacity="0.4" />
          </svg>
        </div>
        <div className="generation-progress__text">
          <div className="generation-progress__title">Generating Storyboard</div>
          <div className="generation-progress__status">{statusText}</div>
        </div>
        <div className="generation-progress__percentage">{job.progress}%</div>
      </div>

      <div className="generation-progress__bar-container">
        <div
          className="generation-progress__bar"
          style={{ width: `${job.progress}%` }}
        />
      </div>

      <div className="generation-progress__meta">
        {job.numFrames} frames · {job.prompt}
      </div>
    </div>
  );
};
```

#### 4.2 Styles (`/frontend/src/styles.css`)

**Add to existing styles:**

```css
/* Generation Progress Component */
.generation-progress {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  padding: 1.25rem 1.5rem;
  border-radius: 1rem;
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(14, 165, 233, 0.08));
  border: 1px solid rgba(59, 130, 246, 0.3);
  max-width: min(650px, 90%);
  align-self: flex-start;
}

.generation-progress__header {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.generation-progress__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 3rem;
  height: 3rem;
  border-radius: 0.75rem;
  background: rgba(59, 130, 246, 0.2);
  color: #60a5fa;
}

.generation-progress__icon--spinning {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.05);
  }
}

.generation-progress__text {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.generation-progress__title {
  font-size: 0.95rem;
  font-weight: 600;
  color: #f8fafc;
}

.generation-progress__status {
  font-size: 0.85rem;
  color: #94a3c3;
}

.generation-progress__percentage {
  font-size: 1.25rem;
  font-weight: 600;
  color: #60a5fa;
  font-feature-settings: 'tnum';
}

.generation-progress__bar-container {
  position: relative;
  width: 100%;
  height: 0.5rem;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.8);
  overflow: hidden;
}

.generation-progress__bar {
  position: absolute;
  top: 0;
  left: 0;
  height: 100%;
  background: linear-gradient(90deg, #3b82f6, #0ea5e9);
  border-radius: 999px;
  transition: width 0.3s ease;
  box-shadow: 0 0 12px rgba(59, 130, 246, 0.6);
}

.generation-progress__bar::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.3),
    transparent
  );
  animation: shimmer 2s infinite;
}

@keyframes shimmer {
  0% {
    transform: translateX(-100%);
  }
  100% {
    transform: translateX(100%);
  }
}

.generation-progress__meta {
  font-size: 0.8rem;
  color: #6d7ba6;
  padding-top: 0.25rem;
  border-top: 1px solid rgba(79, 110, 180, 0.2);
}
```

#### 4.3 Integration with Chat (`/frontend/src/App.tsx`)

**Modify App.tsx to add generation UI:**

```typescript
import { useState } from 'react';
import { ChatInput } from './components/ChatInput';
import { ChatSidebar } from './components/ChatSidebar';
import { MessageList } from './components/MessageList';
import { GenerationProgress } from './components/GenerationProgress';
import { useChat } from './hooks/useChat';
import { useJobPolling } from './hooks/useJobPolling';
import type { StoryboardJob } from './types';

const App = () => {
  const { chats, activeChatId, messages, isLoading, error, sendMessage, startNewChat, selectChat, generateStoryboard } =
    useChat();

  const [currentJob, setCurrentJob] = useState<StoryboardJob | null>(null);

  const { job: pollingJob, isPolling } = useJobPolling({
    jobId: currentJob?.id ?? null,
    onComplete: (completedJob) => {
      console.log('Generation complete:', completedJob);
      // TODO: Save storyboard as message (Task 6)
      setCurrentJob(null);
    },
    onError: (errorMsg) => {
      console.error('Generation failed:', errorMsg);
      setCurrentJob(null);
    },
  });

  const handleGenerate = async (prompt: string) => {
    try {
      const job = await generateStoryboard(prompt, 8);
      setCurrentJob(job);
    } catch (err) {
      console.error('Failed to start generation:', err);
    }
  };

  return (
    <div className="app-shell">
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        isBusy={isLoading || isPolling}
        onNewChat={() => {
          void startNewChat().catch(() => undefined);
        }}
        onSelectChat={(chatId) => {
          void selectChat(chatId).catch(() => undefined);
        }}
      />

      <main className="chat-panel">
        <header className="chat-panel__header">
          <div>
            <h1>Aldar Kose Assistant</h1>
            <p>Generate cinematic storyboards from your prompts</p>
          </div>
          <button
            type="button"
            onClick={() => {
              void startNewChat().catch(() => undefined);
            }}
            disabled={isLoading || isPolling}
          >
            Clear chat
          </button>
        </header>

        <MessageList messages={messages} isLoading={isLoading} />

        {/* Show generation progress */}
        {pollingJob && (
          <div style={{ padding: '0 2rem' }}>
            <GenerationProgress job={pollingJob} />
          </div>
        )}

        {error && (
          <div className="chat-panel__error" role="alert">
            {error}
          </div>
        )}

        <ChatInput
          onSend={sendMessage}
          isLoading={isLoading || isPolling}
          onGenerate={handleGenerate}
        />
      </main>
    </div>
  );
};

export default App;
```

#### 4.4 Enhanced Chat Input (`/frontend/src/components/ChatInput.tsx`)

**Modify to add "Generate Storyboard" button:**

```typescript
import { KeyboardEvent, useRef, useState } from 'react';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading: boolean;
  onGenerate?: (prompt: string) => void;
}

export const ChatInput = ({ onSend, isLoading, onGenerate }: ChatInputProps) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (!trimmed || isLoading) return;

    onSend(trimmed);
    setValue('');
    textareaRef.current?.focus();
  };

  const handleGenerate = () => {
    const trimmed = value.trim();
    if (!trimmed || isLoading || !onGenerate) return;

    onGenerate(trimmed);
    setValue('');
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-input">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Describe your storyboard scene..."
        disabled={isLoading}
        rows={1}
      />
      {onGenerate && (
        <button
          type="button"
          onClick={handleGenerate}
          disabled={isLoading || !value.trim()}
          className="chat-input__generate-btn"
        >
          Generate
        </button>
      )}
      <button type="button" onClick={handleSubmit} disabled={isLoading || !value.trim()}>
        Send
      </button>
    </div>
  );
};
```

**Add button styles to `/frontend/src/styles.css`:**

```css
.chat-input__generate-btn {
  align-self: flex-end;
  padding: 0.85rem 1.25rem;
  border-radius: 0.9rem;
  border: 1px solid rgba(132, 94, 247, 0.5);
  background: linear-gradient(135deg, rgba(132, 94, 247, 0.25), rgba(79, 110, 241, 0.15));
  color: #e2e8f0;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.15s ease, background 0.15s ease;
}

.chat-input__generate-btn:disabled {
  cursor: not-allowed;
  opacity: 0.6;
}

.chat-input__generate-btn:not(:disabled):hover {
  transform: translateY(-1px);
  background: linear-gradient(135deg, rgba(132, 94, 247, 0.35), rgba(79, 110, 241, 0.25));
}
```

### Acceptance Criteria
- [ ] GenerationProgress component shows real-time progress (0-100%)
- [ ] Status text updates based on progress
- [ ] Animated progress bar with shimmer effect
- [ ] Film reel icon pulses during generation
- [ ] "Generate" button added to chat input
- [ ] Progress UI appears when job is active

---

## TASK 5: Frontend - Cinematic Carousel Component

### Context
**Current State:**
- Images currently shown in grid: `/frontend/src/components/MessageBubble.tsx`
- Grid CSS: `.message-bubble__attachments` with `grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))`
- Each image wrapped in `<figure>` with caption
- Styles in `/frontend/src/styles.css`

### Objective
Create full-screen cinematic carousel component with smooth transitions, keyboard navigation, and professional presentation. This is the "wow factor" for judges.

### Technical Specifications

#### 5.1 Install Carousel Library

**Add to `/frontend/package.json`:**
```json
{
  "dependencies": {
    "swiper": "^11.0.5"
  }
}
```

Then run: `npm install`

#### 5.2 Carousel Component (`/frontend/src/components/StoryboardCarousel.tsx` - NEW FILE)

```typescript
import { useEffect, useCallback } from 'react';
import { Swiper, SwiperSlide } from 'swiper/react';
import { Navigation, Pagination, Keyboard, EffectFade } from 'swiper/modules';
import type { Swiper as SwiperType } from 'swiper';

// Import Swiper styles
import 'swiper/css';
import 'swiper/css/navigation';
import 'swiper/css/pagination';
import 'swiper/css/effect-fade';

interface StoryboardFrame {
  url: string;
  caption: string;
}

interface StoryboardCarouselProps {
  frames: StoryboardFrame[];
  isOpen: boolean;
  onClose: () => void;
  initialSlide?: number;
}

export const StoryboardCarousel = ({
  frames,
  isOpen,
  onClose,
  initialSlide = 0,
}: StoryboardCarouselProps) => {
  // Close on ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEsc);
      document.body.style.overflow = 'hidden'; // Prevent background scroll
    }

    return () => {
      document.removeEventListener('keydown', handleEsc);
      document.body.style.overflow = '';
    };
  }, [isOpen, onClose]);

  const handleSlideChange = useCallback((swiper: SwiperType) => {
    console.log('Current slide:', swiper.activeIndex + 1, '/', frames.length);
  }, [frames.length]);

  if (!isOpen) return null;

  return (
    <div className="storyboard-carousel-overlay" onClick={onClose}>
      <div className="storyboard-carousel" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="storyboard-carousel__header">
          <div className="storyboard-carousel__title">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <rect x="2" y="6" width="4" height="12" rx="1" fill="currentColor" />
              <rect x="7" y="4" width="4" height="16" rx="1" fill="currentColor" />
              <rect x="12" y="6" width="4" height="12" rx="1" fill="currentColor" />
              <rect x="17" y="8" width="4" height="8" rx="1" fill="currentColor" />
            </svg>
            <span>Storyboard</span>
          </div>
          <button
            className="storyboard-carousel__close"
            onClick={onClose}
            aria-label="Close carousel"
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Swiper Carousel */}
        <Swiper
          modules={[Navigation, Pagination, Keyboard, EffectFade]}
          effect="fade"
          fadeEffect={{ crossFade: true }}
          speed={600}
          navigation
          pagination={{
            type: 'fraction',
            formatFractionCurrent: (number) => String(number).padStart(2, '0'),
            formatFractionTotal: (number) => String(number).padStart(2, '0'),
          }}
          keyboard={{ enabled: true }}
          initialSlide={initialSlide}
          onSlideChange={handleSlideChange}
          className="storyboard-carousel__swiper"
        >
          {frames.map((frame, index) => (
            <SwiperSlide key={frame.url}>
              <div className="storyboard-carousel__slide">
                <div className="storyboard-carousel__image-container">
                  <img
                    src={frame.url}
                    alt={frame.caption}
                    className="storyboard-carousel__image"
                  />
                </div>
                <div className="storyboard-carousel__caption">
                  <div className="storyboard-carousel__frame-number">
                    Frame {String(index + 1).padStart(2, '0')}
                  </div>
                  <div className="storyboard-carousel__caption-text">{frame.caption}</div>
                </div>
              </div>
            </SwiperSlide>
          ))}
        </Swiper>

        {/* Instructions */}
        <div className="storyboard-carousel__instructions">
          <kbd>←</kbd>
          <kbd>→</kbd>
          <span>Navigate frames</span>
          <span className="storyboard-carousel__divider">•</span>
          <kbd>ESC</kbd>
          <span>Close</span>
        </div>
      </div>
    </div>
  );
};
```

#### 5.3 Carousel Styles (`/frontend/src/styles.css`)

**Add comprehensive carousel styles:**

```css
/* Storyboard Carousel */
.storyboard-carousel-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.95);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.storyboard-carousel {
  position: relative;
  width: 100%;
  max-width: 1400px;
  height: 90vh;
  display: flex;
  flex-direction: column;
  background: rgba(8, 12, 22, 0.8);
  border-radius: 1.5rem;
  overflow: hidden;
  border: 1px solid rgba(79, 110, 180, 0.3);
  backdrop-filter: blur(20px);
}

.storyboard-carousel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.5rem 2rem;
  border-bottom: 1px solid rgba(79, 110, 180, 0.2);
  background: rgba(11, 16, 29, 0.9);
}

.storyboard-carousel__title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 1.125rem;
  font-weight: 600;
  color: #f8fafc;
}

.storyboard-carousel__title svg {
  color: #60a5fa;
}

.storyboard-carousel__close {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 0.5rem;
  border: 1px solid rgba(148, 163, 245, 0.3);
  background: rgba(26, 33, 58, 0.8);
  color: #e2e8f0;
  cursor: pointer;
  transition: background 0.2s ease, transform 0.15s ease;
}

.storyboard-carousel__close:hover {
  background: rgba(67, 56, 202, 0.6);
  transform: scale(1.05);
}

.storyboard-carousel__swiper {
  flex: 1;
  width: 100%;
}

.storyboard-carousel__slide {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 2rem;
}

.storyboard-carousel__image-container {
  position: relative;
  max-width: 100%;
  max-height: 70vh;
  border-radius: 1rem;
  overflow: hidden;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.storyboard-carousel__image {
  display: block;
  max-width: 100%;
  max-height: 70vh;
  width: auto;
  height: auto;
  object-fit: contain;
  animation: kenBurns 15s ease-in-out infinite alternate;
}

/* Subtle Ken Burns effect */
@keyframes kenBurns {
  0% {
    transform: scale(1) translateX(0);
  }
  100% {
    transform: scale(1.05) translateX(5px);
  }
}

.storyboard-carousel__caption {
  margin-top: 2rem;
  text-align: center;
  max-width: 600px;
}

.storyboard-carousel__frame-number {
  font-size: 0.875rem;
  font-weight: 600;
  color: #60a5fa;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}

.storyboard-carousel__caption-text {
  font-size: 1.125rem;
  color: #e2e8f0;
  line-height: 1.6;
}

.storyboard-carousel__instructions {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  padding: 1.25rem;
  background: rgba(11, 16, 29, 0.9);
  border-top: 1px solid rgba(79, 110, 180, 0.2);
  font-size: 0.875rem;
  color: #94a3c3;
}

.storyboard-carousel__instructions kbd {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 2rem;
  padding: 0.25rem 0.5rem;
  border-radius: 0.375rem;
  border: 1px solid rgba(148, 163, 245, 0.3);
  background: rgba(26, 33, 58, 0.8);
  color: #e2e8f0;
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  font-size: 0.8125rem;
  font-weight: 600;
}

.storyboard-carousel__divider {
  color: rgba(148, 163, 245, 0.4);
}

/* Swiper customization */
.storyboard-carousel__swiper .swiper-button-prev,
.storyboard-carousel__swiper .swiper-button-next {
  color: #f8fafc;
  background: rgba(26, 33, 58, 0.85);
  width: 3rem;
  height: 3rem;
  border-radius: 50%;
  border: 1px solid rgba(148, 163, 245, 0.3);
  transition: background 0.2s ease, transform 0.15s ease;
}

.storyboard-carousel__swiper .swiper-button-prev:hover,
.storyboard-carousel__swiper .swiper-button-next:hover {
  background: rgba(67, 56, 202, 0.85);
  transform: scale(1.1);
}

.storyboard-carousel__swiper .swiper-button-prev::after,
.storyboard-carousel__swiper .swiper-button-next::after {
  font-size: 1.25rem;
}

.storyboard-carousel__swiper .swiper-pagination {
  bottom: 1.5rem;
}

.storyboard-carousel__swiper .swiper-pagination-fraction {
  font-size: 1.125rem;
  font-weight: 600;
  color: #f8fafc;
  background: rgba(26, 33, 58, 0.9);
  padding: 0.5rem 1rem;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 245, 0.3);
  width: auto;
  left: 50%;
  transform: translateX(-50%);
}

/* Mobile responsive */
@media (max-width: 768px) {
  .storyboard-carousel-overlay {
    padding: 0;
  }

  .storyboard-carousel {
    height: 100vh;
    border-radius: 0;
  }

  .storyboard-carousel__slide {
    padding: 1rem;
  }

  .storyboard-carousel__image-container {
    max-height: 60vh;
  }

  .storyboard-carousel__instructions {
    display: none;
  }
}
```

#### 5.4 Enhanced MessageBubble (`/frontend/src/components/MessageBubble.tsx`)

**Add carousel trigger:**

```typescript
import { useState } from 'react';
import type { ChatMessage, MessageAttachment } from '../types';
import { StoryboardCarousel } from './StoryboardCarousel';

interface MessageBubbleProps {
  message: ChatMessage;
}

const roleLabel: Record<ChatMessage['role'], string> = {
  user: 'You',
  assistant: 'Assistant',
};

const renderAttachment = (attachment: MessageAttachment, index: number) => {
  if (attachment.type === 'image') {
    return (
      <figure key={`${attachment.url}-${index}`} className="message-bubble__attachment">
        <img src={attachment.url} alt={attachment.alt} loading="lazy" />
        <figcaption>{attachment.alt}</figcaption>
      </figure>
    );
  }

  return null;
};

export const MessageBubble = ({ message }: MessageBubbleProps) => {
  const [carouselOpen, setCarouselOpen] = useState(false);
  const [initialSlide, setInitialSlide] = useState(0);

  const bubbleClassName = `message-bubble message-bubble--${message.role}`;
  const createdAt = new Date(message.createdAt);
  const displayTime = Number.isNaN(createdAt.getTime())
    ? message.createdAt
    : createdAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const hasAttachments = message.attachments && message.attachments.length > 0;
  const isStoryboard = hasAttachments && message.attachments.length >= 6;

  const handleImageClick = (index: number) => {
    setInitialSlide(index);
    setCarouselOpen(true);
  };

  const frames = message.attachments?.map((att) => ({
    url: att.url,
    caption: att.alt,
  })) || [];

  return (
    <>
      <article className={bubbleClassName} aria-label={`${roleLabel[message.role]} message`}>
        <header className="message-bubble__meta">
          <span className="message-bubble__author">{roleLabel[message.role]}</span>
          <time dateTime={message.createdAt}>{displayTime}</time>
        </header>
        <div className="message-bubble__content">
          {message.content.split('\n').map((line, index) => (
            <p key={index}>{line}</p>
          ))}
        </div>
        {hasAttachments && (
          <>
            {isStoryboard && (
              <button
                className="message-bubble__view-storyboard"
                onClick={() => setCarouselOpen(true)}
              >
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect x="2" y="6" width="4" height="12" rx="1" fill="currentColor" />
                  <rect x="7" y="4" width="4" height="16" rx="1" fill="currentColor" />
                  <rect x="12" y="6" width="4" height="12" rx="1" fill="currentColor" />
                  <rect x="17" y="8" width="4" height="8" rx="1" fill="currentColor" />
                </svg>
                View Storyboard ({message.attachments.length} frames)
              </button>
            )}
            <div className="message-bubble__attachments" aria-label="Attachments">
              {message.attachments.map((attachment, index) => (
                <div
                  key={`${attachment.url}-${index}`}
                  onClick={() => handleImageClick(index)}
                  style={{ cursor: 'pointer' }}
                >
                  {renderAttachment(attachment, index)}
                </div>
              ))}
            </div>
          </>
        )}
      </article>

      {isStoryboard && (
        <StoryboardCarousel
          frames={frames}
          isOpen={carouselOpen}
          onClose={() => setCarouselOpen(false)}
          initialSlide={initialSlide}
        />
      )}
    </>
  );
};
```

**Add button styles:**

```css
.message-bubble__view-storyboard {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
  padding: 0.6rem 1rem;
  border-radius: 0.65rem;
  border: 1px solid rgba(59, 130, 246, 0.4);
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(14, 165, 233, 0.15));
  color: #60a5fa;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.15s ease, background 0.15s ease;
}

.message-bubble__view-storyboard:hover {
  transform: translateY(-1px);
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(14, 165, 233, 0.2));
  box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
}

.message-bubble__view-storyboard svg {
  color: #60a5fa;
}
```

### Acceptance Criteria
- [ ] Full-screen carousel with dark overlay
- [ ] Smooth fade transitions between frames
- [ ] Ken Burns subtle animation on images
- [ ] Navigation arrows and keyboard support (←/→, ESC)
- [ ] Frame counter (01/08 format)
- [ ] Captions below images
- [ ] "View Storyboard" button on messages with 6+ images
- [ ] Click thumbnail to open at that frame

---

## TASK 6: Complete Integration & Save Storyboards

### Context
**All components built, need to connect them.**

### Objective
Wire everything together: when job completes, save storyboard as chat message.

### Technical Specifications

#### 6.1 Complete App.tsx Integration

**Update the onComplete handler in `/frontend/src/App.tsx`:**

```typescript
import type { MessageAttachment } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api/v1').replace(/\/+$/, '');

const { job: pollingJob, isPolling } = useJobPolling({
  jobId: currentJob?.id ?? null,
  onComplete: async (completedJob) => {
    console.log('Generation complete:', completedJob);

    if (!completedJob.resultUrls || completedJob.resultUrls.length === 0) {
      console.warn('Job completed but no results');
      setCurrentJob(null);
      return;
    }

    // Create assistant message with storyboard
    const attachments: MessageAttachment[] = completedJob.resultUrls.map((url, index) => ({
      type: 'image',
      url: url,
      alt: `Frame ${index + 1}: ${completedJob.prompt}`,
    }));

    const messagePayload = {
      role: 'assistant',
      content: `Generated ${completedJob.numFrames}-frame storyboard for: "${completedJob.prompt}"`,
      attachments: attachments,
    };

    try {
      // Save to backend
      const response = await fetch(
        `${API_BASE}/chats/${completedJob.chatId}/messages`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(messagePayload),
        }
      );

      if (!response.ok) {
        throw new Error('Failed to save storyboard message');
      }

      // Reload messages to show storyboard
      await selectChat(completedJob.chatId);
      setCurrentJob(null);
    } catch (err) {
      console.error('Failed to save storyboard:', err);
      setError('Generated storyboard but failed to save. Please refresh.');
      setCurrentJob(null);
    }
  },
  onError: (errorMsg) => {
    console.error('Generation failed:', errorMsg);
    setError(`Storyboard generation failed: ${errorMsg}`);
    setCurrentJob(null);
  },
});
```

### Acceptance Criteria
- [ ] Completed storyboards saved as assistant messages
- [ ] Messages appear in chat history
- [ ] "View Storyboard" button shows on saved messages
- [ ] Can generate multiple storyboards in same chat
- [ ] Errors handled gracefully

---

## TASK 7: Docker Deployment Setup

### Context
**All code ready, need deployment configuration.**

### Objective
Create Docker setup for easy deployment of all services.

### Technical Specifications

#### 7.1 Docker Compose (`/docker-compose.yml`)

**Create/update docker-compose.yml:**

```yaml
version: '3.9'

services:
  # Backend API
  backend:
    build:
      context: ./backend
    container_name: aldar-kose-backend
    environment:
      - DATABASE_URL=sqlite:///data/app.db
      - GPU_SERVICE_URL=http://gpu-service:8001
      - BACKEND_CALLBACK_URL=http://backend:8000/api/v1
      - IMAGES_DIR=/app/images
    volumes:
      - ./backend/data:/app/data
      - shared-images:/app/images
    ports:
      - '8000:8000'
    networks:
      - aldar-kose-network
    healthcheck:
      test: curl -f http://localhost:8000/health || exit 1
      interval: 10s
      timeout: 5s
      retries: 3

  # Frontend
  frontend:
    build:
      context: ./frontend
    container_name: aldar-kose-frontend
    environment:
      - VITE_USE_MOCK=false
      - VITE_API_BASE=/api/v1
      - VITE_PROXY_TARGET=http://backend:8000
    ports:
      - '5173:5173'
    depends_on:
      - backend
    networks:
      - aldar-kose-network

  # GPU Inference Service (deploy separately on GPU server)
  # Uncomment if running GPU on same host
  # gpu-service:
  #   build:
  #     context: ./ml
  #     dockerfile: Dockerfile.service
  #   container_name: aldar-kose-gpu
  #   environment:
  #     - OUTPUT_DIR=/app/outputs
  #     - IMAGE_BASE_URL=http://localhost:8000/images
  #   volumes:
  #     - shared-images:/app/outputs
  #     - ./ml/configs:/app/configs:ro
  #   ports:
  #     - '8001:8001'
  #   deploy:
  #     resources:
  #       reservations:
  #         devices:
  #           - driver: nvidia
  #             count: 1
  #             capabilities: [gpu]
  #   networks:
  #     - aldar-kose-network

volumes:
  shared-images:
    driver: local

networks:
  aldar-kose-network:
    driver: bridge
```

#### 7.2 GPU Service Deployment (Separate Server)

**If GPU is on separate server, use this setup:**

```yaml
# gpu-server/docker-compose.yml
version: '3.9'

services:
  gpu-service:
    build:
      context: .
      dockerfile: Dockerfile.service
    container_name: aldar-kose-gpu
    environment:
      - OUTPUT_DIR=/app/outputs
      - IMAGE_BASE_URL=${BACKEND_URL}/images
      - PORT=8001
    volumes:
      - ./outputs:/app/outputs
      - ./configs:/app/configs:ro
    ports:
      - '8001:8001'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

**Then configure backend to point to GPU server:**
```bash
# In backend .env or docker-compose
GPU_SERVICE_URL=http://your-gpu-server-ip:8001
```

#### 7.3 Environment Variables

**Create `.env.example`:**

```env
# Backend
DATABASE_URL=sqlite:///data/app.db
GPU_SERVICE_URL=http://gpu-service:8001
BACKEND_CALLBACK_URL=http://backend:8000/api/v1
IMAGES_DIR=/app/images

# Frontend
VITE_USE_MOCK=false
VITE_API_BASE=/api/v1
VITE_PROXY_TARGET=http://backend:8000

# GPU Service
OUTPUT_DIR=/app/outputs
IMAGE_BASE_URL=http://localhost:8000/images
PORT=8001
```

### Acceptance Criteria
- [ ] `docker-compose up` starts backend + frontend
- [ ] GPU service can run separately on GPU server
- [ ] Shared volume for images works correctly
- [ ] Health checks pass for all services
- [ ] Environment variables configurable

### Testing

```bash
# Local (without GPU service)
docker-compose up --build

# With GPU on same machine
# Uncomment gpu-service in docker-compose.yml
docker-compose up --build

# GPU on separate server
# On GPU server:
cd ml
./deploy_gpu_service.sh

# On main server:
export GPU_SERVICE_URL=http://gpu-server-ip:8001
docker-compose up backend frontend
```

---

## TASK 8: Demo Preparation & Documentation

### Context
**System complete, prepare for hackathon presentation.**

### Objective
Create demo script, test end-to-end flow, prepare fallback plans.

### Technical Specifications

#### 8.1 Demo Preparation Checklist

**Pre-demo (15 minutes before):**

1. **Clean environment:**
```bash
rm -rf backend/data/app.db
docker-compose down -v
```

2. **Start services:**
```bash
docker-compose up -d
# Wait 30 seconds for startup
docker-compose ps  # Verify all running
```

3. **Test generation:**
- Open http://localhost:5173
- Generate test storyboard with 6 frames
- Verify carousel works
- Clear chat for demo

4. **Prepare prompts:**
```
1. "Aldar Kose tricks a greedy merchant by selling him a magic pot at the bazaar"
2. "The trickster hero escapes on horseback across the endless Kazakh steppe at sunset"
3. "Aldar Kose disguises himself as a wise elder to fool the greedy bai"
```

#### 8.2 Demo Script (4 minutes)

**Opening (30 sec):**
> "Aldar Kose is a legendary trickster from Kazakh folklore. We built an AI system that transforms his tales into cinematic storyboards in real-time."

**Architecture (45 sec):**
> "We use a microservice architecture: React frontend, FastAPI backend with SQLite for job tracking, and a GPU inference service running our diffusion models. The services communicate via REST APIs - production-ready and horizontally scalable."

**Live Demo (2 min):**
1. Show UI: "ChatGPT-like interface"
2. Enter prompt: [Use prepared prompt #1]
3. Click "Generate Storyboard"
4. While generating: "Real-time progress tracking - AI is planning shots, then generating each frame with GPU acceleration"
5. When complete: Click "View Storyboard"
6. Navigate carousel: "Full-screen cinematic presentation with smooth transitions"
7. Navigate frames: Show 2-3 frames

**Technical Highlight (30 sec):**
> "FastAPI for async job management, stateless GPU microservice for scalability, Swiper.js for cinematic UX. All containerized with Docker. The same pipeline works for any story from any culture."

**Close (15 sec):**
> "This demonstrates how AI can preserve and visualize cultural narratives. Thank you!"

#### 8.3 Troubleshooting

**GPU service not responding:**
```bash
# Check GPU service logs
docker logs aldar-kose-gpu

# Restart GPU service
docker restart aldar-kose-gpu

# Fallback: Use mock mode
export VITE_USE_MOCK=true
docker-compose up frontend
```

**Images not loading:**
```bash
# Check shared volume
docker exec aldar-kose-backend ls -la /app/images

# Check backend logs
docker logs aldar-kose-backend | grep images
```

**Demo fallback plan:**
- Pre-record video of full generation flow
- Have screenshots of carousel ready
- Can walk through architecture diagram if live demo fails

#### 8.4 README Update

**Update `/README.md`:**

```markdown
# Aldar Kose Storyboard Generator

AI-powered cinematic storyboard generation from text prompts. Built for HackNU 2025.

## Architecture

```
Frontend (React) → Backend (FastAPI) → GPU Service (Microservice)
                     ↓
                 SQLite Jobs DB
```

**Production-ready microservice pattern:**
- Stateless GPU inference service (scalable)
- Async job tracking with status polling
- RESTful API design

## Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for GPU service)

### Run

```bash
# Start backend + frontend
docker-compose up --build

# Deploy GPU service (on GPU server)
cd ml
./deploy_gpu_service.sh

# Set GPU service URL
export GPU_SERVICE_URL=http://your-gpu-server:8001

# Open browser
open http://localhost:5173
```

### Services

- Frontend: http://localhost:5173
- Backend: http://localhost:8000/docs
- GPU Service: http://localhost:8001/health

## Usage

1. Enter scene description
2. Click "Generate Storyboard"
3. Watch real-time progress
4. View cinematic presentation

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite, Swiper.js
- **Backend**: FastAPI, SQLAlchemy, SQLite, httpx
- **ML**: Stable Diffusion, IP-Adapter, LLM planning
- **Deployment**: Docker, nvidia-docker

## Architecture Highlights

- Microservice pattern (GPU service can scale independently)
- Async job processing with real-time status updates
- Stateless inference service (easy to load balance)
- Production-ready error handling and health checks

## License

MIT

## Credits

HackNU 2025 - Built with Claude Code
```

### Acceptance Criteria
- [ ] End-to-end flow tested
- [ ] Demo script under 5 minutes
- [ ] Prepared prompts generate good storyboards
- [ ] Fallback plan ready
- [ ] README updated with architecture

---

## Summary: Implementation Timeline

**Estimated time: 3-4 hours**

| Task | Time | Priority |
|------|------|----------|
| 1. Backend Job System | 30 min | CRITICAL |
| 2. GPU Microservice | 30 min | CRITICAL |
| 3. Frontend Job Submission | 30 min | CRITICAL |
| 4. Generation Progress UI | 45 min | HIGH |
| 5. Cinematic Carousel | 60 min | HIGH |
| 6. Complete Integration | 15 min | CRITICAL |
| 7. Docker Setup | 20 min | MEDIUM |
| 8. Demo Prep | 20 min | MEDIUM |

**Total: ~3.5 hours**

## Critical Path (Minimum Viable Demo)

**Must-have (2.5 hours):**
1. Tasks 1-3: Backend + GPU + Frontend submission
2. Task 4: Progress UI
3. Task 5: Carousel
4. Task 6: Save storyboards

**Nice-to-have:**
- Task 7: Docker polish
- Task 8: Documentation

## Success Metrics

- ✅ User submits prompt, sees real-time progress
- ✅ GPU generates 6-10 frames
- ✅ Cinematic carousel impresses judges
- ✅ Complete demo under 5 minutes
- ✅ No critical errors during presentation

---

**Microservice architecture advantages for hackathon:**
- Simpler code (no RabbitMQ complexity)
- GPU runs independently (easier debugging)
- Production-ready pattern (impresses AI company judges)
- Faster implementation (3.5 hrs vs 6+ hrs)
- Better demo narrative ("microservice architecture")

**Good luck with HackNU 2025!** 🚀