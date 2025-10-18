from __future__ import annotations

import asyncio
import logging
import os
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ... import crud, gpu_client, schemas
from ...database import SessionLocal, get_db

router = APIRouter()
GPU_CALLBACK_BASE = os.getenv("GPU_CALLBACK_BASE", "http://localhost:9000/api/v1")
logger = logging.getLogger(__name__)


@router.post("/chats/{chat_id}/jobs", response_model=schemas.JobRead, status_code=201)
async def create_job(
    chat_id: str,
    job_in: schemas.JobCreate,
    db: Session = Depends(get_db),
):
    try:
        job = crud.create_job(db, chat_id=chat_id, job_in=job_in)
    except crud.ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    logger.info("Scheduling GPU generation for job %s (chat %s)", job.id, chat_id)
    asyncio.create_task(process_job(str(job.id), job_in.prompt, job_in.num_frames))

    return job


@router.get("/jobs/{job_id}", response_model=schemas.JobRead)
def get_job(job_id: str, db: Session = Depends(get_db)):
    try:
        job = crud.get_job(db, job_id=job_id)
    except crud.JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return job


@router.get("/chats/{chat_id}/jobs", response_model=List[schemas.JobRead])
def list_jobs(chat_id: str, db: Session = Depends(get_db)):
    try:
        jobs = crud.list_jobs_by_chat(db, chat_id=chat_id)
    except crud.ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return jobs


@router.patch("/jobs/{job_id}", response_model=schemas.JobRead)
def update_job(
    job_id: str,
    job_update: schemas.JobUpdate,
    db: Session = Depends(get_db),
):
    try:
        job = crud.update_job(db, job_id=job_id, job_update=job_update)
    except crud.JobNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return job


async def process_job(job_id: str, prompt: str, num_frames: int):
    db = SessionLocal()
    try:
        job_update = schemas.JobUpdate(status="processing", progress=10)
        crud.update_job(db, job_id=job_id, job_update=job_update)

        callback_url = GPU_CALLBACK_BASE.rstrip("/")

        logger.info(
            "Submitting job %s to GPU service at %s (callback %s)",
            job_id,
            gpu_client.GPU_SERVICE_URL,
            callback_url,
        )
        await gpu_client.submit_job(
            prompt=prompt,
            num_frames=num_frames,
            job_id=job_id,
            callback_url=callback_url,
        )

        logger.info("Job %s successfully handed off to GPU service", job_id)

    except Exception as e:
        job_update = schemas.JobUpdate(status="failed", error_message=str(e))
        crud.update_job(db, job_id=job_id, job_update=job_update)
        logger.exception("GPU job %s failed to submit", job_id)
    finally:
        db.close()
