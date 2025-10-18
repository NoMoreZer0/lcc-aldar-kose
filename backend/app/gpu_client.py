from __future__ import annotations

import logging
import os
from typing import Optional

import httpx


class GPUServiceError(Exception):
    pass


GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://host.docker.internal:8081")
DEFAULT_TIMEOUT = 30.0
logger = logging.getLogger(__name__)


async def submit_job(
    prompt: str,
    num_frames: int,
    job_id: Optional[str] = None,
    callback_url: Optional[str] = None,
) -> dict:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            logger.info(
                "Calling GPU service %s/generate for job %s (%s frames)",
                GPU_SERVICE_URL,
                job_id or "<unknown>",
                num_frames,
            )
            response = await client.post(
                f"{GPU_SERVICE_URL}/generate",
                json={
                    "job_id": job_id,
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "callback_url": callback_url,
                },
            )
            response.raise_for_status()
            payload = response.json()
            logger.info(
                "GPU service accepted job %s with status %s: %s",
                job_id or "<unknown>",
                response.status_code,
                payload,
            )
            return payload
        except httpx.HTTPStatusError as e:
            logger.error(
                "GPU service responded with HTTP %s for job %s: %s",
                e.response.status_code,
                job_id or "<unknown>",
                e.response.text,
            )
            raise GPUServiceError(f"GPU service returned error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            logger.error(
                "Failed to reach GPU service at %s for job %s: %s",
                GPU_SERVICE_URL,
                job_id or "<unknown>",
                e,
            )
            raise GPUServiceError(f"Failed to connect to GPU service: {str(e)}") from e


async def get_status(job_id: str) -> dict:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.get(f"{GPU_SERVICE_URL}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise GPUServiceError(f"GPU service returned error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise GPUServiceError(f"Failed to connect to GPU service: {str(e)}") from e
