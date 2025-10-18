from fastapi import APIRouter

from .v1 import chats

api_router = APIRouter()
api_router.include_router(chats.router, prefix="/v1")
