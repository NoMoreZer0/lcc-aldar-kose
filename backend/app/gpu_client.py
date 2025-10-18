from __future__ import annotations

import os
from typing import Optional

import httpx


class GPUServiceError(Exception):
    pass


GPU_SERVICE_URL = os.getenv("GPU_SERVICE_URL", "http://localhost:8001")
DEFAULT_TIMEOUT = 30.0


async def submit_job(
    prompt: str,
    num_frames: int,
    callback_url: Optional[str] = None,
) -> dict:
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            response = await client.post(
                f"{GPU_SERVICE_URL}/generate",
                json={
                    "prompt": prompt,
                    "num_frames": num_frames,
                    "callback_url": callback_url,
                },
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise GPUServiceError(f"GPU service returned error: {e.response.status_code}") from e
        except httpx.RequestError as e:
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
