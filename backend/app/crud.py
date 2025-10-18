from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.orm import Session, joinedload

from . import schemas
from .models import AttachmentType, Chat, Message, MessageAttachment, MessageRole


class ChatNotFoundError(Exception):
    pass


def create_chat(db: Session, chat_in: schemas.ChatCreate) -> Chat:
    chat = Chat(title=chat_in.title)
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def list_chats(db: Session) -> List[schemas.ChatSummary]:
    stmt: Select = (
        select(
            Chat,
            func.count(Message.id).label("message_count"),
            func.max(Message.created_at).label("last_message_at"),
        )
        .outerjoin(Message)
        .group_by(Chat.id)
        .order_by(func.max(Message.created_at).desc().nullslast(), Chat.created_at.desc())
    )

    results = db.execute(stmt).all()

    summaries: List[schemas.ChatSummary] = []
    for chat, message_count, last_message_at in results:
        summaries.append(
            schemas.ChatSummary(
                id=UUID(chat.id),
                title=chat.title,
                created_at=chat.created_at,
                updated_at=chat.updated_at,
                message_count=int(message_count or 0),
                last_message_at=last_message_at,
            )
        )

    return summaries


def get_chat(db: Session, chat_id: str) -> Chat:
    chat = (
        db.query(Chat)
        .options(joinedload(Chat.messages).joinedload(Message.attachments))
        .filter(Chat.id == chat_id)
        .one_or_none()
    )

    if chat is None:
        raise ChatNotFoundError(f"Chat {chat_id} not found")

    return chat


def create_message(db: Session, chat_id: str, message_in: schemas.MessageCreate) -> Message:
    chat = db.query(Chat).filter(Chat.id == chat_id).one_or_none()
    if chat is None:
        raise ChatNotFoundError(f"Chat {chat_id} not found")

    sequence = (db.query(func.count(Message.id)).filter(Message.chat_id == chat_id).scalar() or 0) + 1

    message = Message(
        chat_id=chat_id,
        role=MessageRole(message_in.role),
        content=message_in.content,
        sequence=sequence,
    )

    attachments = []
    for attachment_in in message_in.attachments or []:
        attachment = MessageAttachment(
            type=AttachmentType(attachment_in.type),
            url=str(attachment_in.url),
            alt=attachment_in.alt,
        )
        attachments.append(attachment)

    if attachments:
        message.attachments.extend(attachments)

    db.add(message)

    # keep chat updated timestamp in sync
    chat.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(message)
    return message


def delete_chat(db: Session, chat_id: str) -> None:
    chat = db.query(Chat).filter(Chat.id == chat_id).one_or_none()
    if chat is None:
        raise ChatNotFoundError(f"Chat {chat_id} not found")

    db.delete(chat)
    db.commit()


def list_messages(db: Session, chat_id: str) -> List[Message]:
    chat = get_chat(db, chat_id)
    return chat.messages
