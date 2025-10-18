from __future__ import annotations

import asyncio
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ... import crud, gpu_client, schemas
from ...database import get_db

router = APIRouter()


@router.post("/chats/{chat_id}/jobs", response_model=schemas.JobRead, status_code=201)
async def create_job(
    chat_id: str,
    job_in: schemas.JobCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        job = crud.create_job(db, chat_id=chat_id, job_in=job_in)
    except crud.ChatNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    background_tasks.add_task(process_job, str(job.id), job_in.prompt, job_in.num_frames, db)

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


async def process_job(job_id: str, prompt: str, num_frames: int, db: Session):
    try:
        job_update = schemas.JobUpdate(status="processing", progress=10)
        crud.update_job(db, job_id=job_id, job_update=job_update)

        result = await gpu_client.submit_job(prompt=prompt, num_frames=num_frames)

        gpu_job_id = result.get("job_id")
        if not gpu_job_id:
            raise ValueError("GPU service did not return job_id")

        while True:
            await asyncio.sleep(2)
            status_result = await gpu_client.get_status(gpu_job_id)

            progress = status_result.get("progress", 0)
            status = status_result.get("status", "processing")

            job_update = schemas.JobUpdate(progress=progress)

            if status == "completed":
                result_urls = status_result.get("result_urls", [])
                job_update.status = "completed"
                job_update.result_urls = result_urls
                crud.update_job(db, job_id=job_id, job_update=job_update)
                break
            elif status == "failed":
                error_message = status_result.get("error_message", "Unknown error")
                job_update.status = "failed"
                job_update.error_message = error_message
                crud.update_job(db, job_id=job_id, job_update=job_update)
                break
            else:
                crud.update_job(db, job_id=job_id, job_update=job_update)

    except Exception as e:
        job_update = schemas.JobUpdate(status="failed", error_message=str(e))
        crud.update_job(db, job_id=job_id, job_update=job_update)
