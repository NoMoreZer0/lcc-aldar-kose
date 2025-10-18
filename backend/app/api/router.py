from fastapi import APIRouter

from .v1 import chats, jobs

api_router = APIRouter()
api_router.include_router(chats.router, prefix="/v1")
api_router.include_router(jobs.router, prefix="/v1")
