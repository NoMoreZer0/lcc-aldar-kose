"""
GPU Inference Microservice for Aldar Kose storyboard generation.

Expose a FastAPI service that accepts generation requests, runs the
storybook pipeline, and reports progress back to the main backend.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import List

import httpx
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, Field

# Ensure project packages are importable when running `python service.py`
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from ml.src.diffusion.pipeline import StoryboardGenerationPipeline
from ml.src.llm.story_context import build_plan_payload, frames_to_shots, generate_story_frames
from ml.src.utils import io as io_utils
from ml.src.utils.determinism import configure_determinism, set_seed
from ml.src.utils.schema import ConfigModel

CONFIG_PATH = CURRENT_DIR / "configs" / "default.yaml"
OUTPUT_BASE = Path(os.getenv("OUTPUT_DIR", CURRENT_DIR / "outputs"))
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "").rstrip("/")
S3_BUCKET = os.getenv("S3_BUCKET", "storyboards")
S3_PREFIX = os.getenv("S3_PREFIX", "").strip("/")
ENABLE_S3_UPLOAD = os.getenv("ENABLE_S3_UPLOAD", "true").lower() != "false"
S3_PRESIGN_URLS = os.getenv("S3_PRESIGN_URLS", "true").lower() != "false"
S3_PRESIGNED_TTL = int(os.getenv("S3_PRESIGNED_TTL", "86400"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gpu_service")

app = FastAPI(title="Aldar Kose GPU Inference Service", version="1.0.0")

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


class GenerationRequest(BaseModel):
    job_id: str = Field(..., description="Unique job identifier from the backend")
    prompt: str = Field(..., min_length=1, max_length=2000, description="Storyboard prompt/logline")
    num_frames: int = Field(default=8, ge=6, le=10, description="Number of storyboard frames to generate")
    callback_url: str = Field(..., description="Backend callback base URL (e.g. http://backend/api/v1)")


class GenerationResponse(BaseModel):
    status: str
    job_id: str
    message: str


async def _report_patch(callback_url: str, job_id: str, payload: dict) -> None:
    """Send a PATCH request to the backend and swallow network failures."""
    target = f"{callback_url.rstrip('/')}/jobs/{job_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(target, json=payload)
    except Exception as exc:  # pragma: no cover - network path
        logger.error("Failed to report update for %s: %s", job_id, exc)


async def report_progress(callback_url: str, job_id: str, progress: int, status: str = "processing") -> None:
    await _report_patch(callback_url, job_id, {"status": status, "progress": progress})
    logger.info("Reported progress %s%% for job %s", progress, job_id)


async def report_completion(callback_url: str, job_id: str, image_urls: List[str]) -> None:
    await _report_patch(
        callback_url,
        job_id,
        {"status": "completed", "progress": 100, "result_urls": image_urls},
    )
    logger.info("Reported completion for job %s", job_id)


async def report_failure(callback_url: str, job_id: str, error_message: str) -> None:
    await _report_patch(
        callback_url,
        job_id,
        {"status": "failed", "progress": 0, "error_message": error_message},
    )
    logger.error("Reported failure for job %s: %s", job_id, error_message)


def _object_prefix(run_id: str) -> str:
    parts = [part for part in (S3_PREFIX, run_id) if part]
    return "/".join(parts)


def _build_image_urls(run_id: str, filenames: List[str]) -> List[str]:
    prefix = _object_prefix(run_id)
    keys = [f"{prefix}/{filename}" if prefix else filename for filename in filenames]

    if ENABLE_S3_UPLOAD and S3_PRESIGN_URLS:
        presigned = io_utils.generate_presigned_urls(
            S3_BUCKET,
            keys,
            expires_in=S3_PRESIGNED_TTL,
        )
        if presigned is not None:
            return presigned
        logger.warning("Presigned URLs unavailable; falling back to static paths for job %s", run_id)

    urls: List[str] = []
    for key in keys:
        if IMAGE_BASE_URL:
            urls.append(f"{IMAGE_BASE_URL}/{key}")
        else:
            urls.append(f"s3://{S3_BUCKET}/{key}")

    return urls


async def run_generation(request: GenerationRequest) -> None:
    """Orchestrate storyboard generation and callback updates."""
    job_id = request.job_id
    logger.info("Starting generation for job %s (%s frames)", job_id, request.num_frames)

    try:
        config_model = ConfigModel.from_path(CONFIG_PATH)
        config_dict = (
            config_model.model_dump()
            if hasattr(config_model, "model_dump")
            else config_model.dict()
        )
        config_dict = config_dict | {
            "consistency": {
                "use_consistory": False,
                "use_img2img": True,
                "img2img_strength": 0.9,
            }
        }

        configure_determinism(True)
        base_seed = set_seed(None)

        await report_progress(request.callback_url, job_id, 10)

        logger.info("Generating narrative plan with GPT-5 for job %s", job_id)
        frames = generate_story_frames(request.prompt, request.num_frames)
        plan_payload = build_plan_payload(request.prompt, frames)
        shots = frames_to_shots(frames)

        await report_progress(request.callback_url, job_id, 30)

        run_id = f"job_{job_id[:8]}_{int(time.time())}"
        output_dir = io_utils.ensure_dir(OUTPUT_BASE / run_id)
        plan_path = output_dir / "plan.json"
        io_utils.dump_json(plan_path, plan_payload)
        logger.info("Saved GPT plan for job %s to %s", job_id, plan_path)

        pipeline = StoryboardGenerationPipeline(config_dict, logger=logger)

        logger.info("Generating frames for job %s into %s", job_id, output_dir)
        index_payload = pipeline.run(
            logline=request.prompt,
            shots=shots,
            output_dir=output_dir,
            use_controlnet=config_model.model.get("use_controlnet", True),
            base_seed=base_seed,
        )

        await report_progress(request.callback_url, job_id, 90)

        index_path = output_dir / "index.json"
        index_dict = (
            index_payload.model_dump()
            if hasattr(index_payload, "model_dump")
            else index_payload.dict()
        )
        io_utils.dump_json(index_path, index_dict)

        frame_files = sorted(output_dir.glob("frame_*.png"))
        filenames = [frame_file.name for frame_file in frame_files]

        if ENABLE_S3_UPLOAD:
            object_prefix = _object_prefix(run_id) or run_id
            logger.info("Uploading results for job %s to s3://%s/%s", job_id, S3_BUCKET, object_prefix)
            uploaded = io_utils.upload_to_s3(output_dir, S3_BUCKET, prefix=object_prefix)
            if uploaded is None:
                logger.warning("S3 upload skipped for job %s; boto3 not available", job_id)

        image_urls = _build_image_urls(run_id, filenames)

        await report_completion(request.callback_url, job_id, image_urls)
    except Exception as exc:  # pragma: no cover - pipeline failure
        logger.exception("Job %s failed", job_id)
        await report_failure(request.callback_url, job_id, str(exc))


@app.post("/generate", response_model=GenerationResponse, status_code=202)
async def generate_storyboard(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
) -> GenerationResponse:
    """Accept a new storyboard generation job."""
    logger.info("Received generation request for job %s", request.job_id)
    background_tasks.add_task(run_generation, request)
    return GenerationResponse(
        status="accepted",
        job_id=request.job_id,
        message=f"Generation started for {request.num_frames} frames",
    )


@app.get("/health")
async def health_check() -> dict:
    """Simple health probe."""
    return {
        "status": "healthy",
        "service": "gpu-inference",
        "gpu_available": check_gpu_available(),
    }


def check_gpu_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:  # pragma: no cover - optional dependency
        return False


@app.get("/")
async def root() -> dict:
    return {
        "service": "Aldar Kose GPU Inference Microservice",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /generate",
            "health": "GET /health",
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual execution
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8002")),
        log_level="info",
    )
