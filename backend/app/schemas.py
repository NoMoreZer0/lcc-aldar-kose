from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field

MessageRole = Literal["user", "assistant"]
AttachmentType = Literal["image"]


class AttachmentBase(BaseModel):
    type: AttachmentType = Field(default="image", description="Attachment type")
    url: AnyHttpUrl = Field(description="Public URL to the attachment resource")
    alt: str = Field(description="Accessible description for the attachment")


class AttachmentCreate(AttachmentBase):
    pass


class AttachmentRead(AttachmentBase):
    id: UUID
    created_at: datetime

    class Config:
        orm_mode = True


class MessageBase(BaseModel):
    role: MessageRole
    content: str


class MessageCreate(MessageBase):
    attachments: Optional[List[AttachmentCreate]] = None


class MessageRead(MessageBase):
    id: UUID
    sequence: int
    created_at: datetime
    attachments: List[AttachmentRead] = Field(default_factory=list)

    class Config:
        orm_mode = True


class ChatBase(BaseModel):
    title: Optional[str] = None


class ChatCreate(ChatBase):
    pass


class ChatUpdate(ChatBase):
    pass


class ChatSummary(ChatBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_at: Optional[datetime]

    class Config:
        orm_mode = True


class ChatRead(ChatBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    messages: List[MessageRead] = Field(default_factory=list)

    class Config:
        orm_mode = True


JobStatusType = Literal["pending", "processing", "completed", "failed"]


class JobCreate(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    num_frames: int = Field(default=8, ge=1, le=10)


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
