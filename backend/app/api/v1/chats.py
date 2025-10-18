from __future__ import annotations

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.orm import Session

from ... import crud, schemas
from ...crud import ChatNotFoundError
from ...database import get_db

router = APIRouter(prefix="/chats", tags=["chats"])


@router.post("", response_model=schemas.ChatRead, status_code=status.HTTP_201_CREATED)
def create_chat(chat_in: schemas.ChatCreate, db: Session = Depends(get_db)) -> schemas.ChatRead:
    chat = crud.create_chat(db, chat_in)
    return schemas.ChatRead.from_orm(crud.get_chat(db, chat.id))


@router.get("", response_model=List[schemas.ChatSummary])
def list_chats(db: Session = Depends(get_db)) -> List[schemas.ChatSummary]:
    return crud.list_chats(db)


@router.get("/{chat_id}", response_model=schemas.ChatRead)
def get_chat(
    chat_id: UUID = Path(..., description="Unique chat identifier"),
    db: Session = Depends(get_db),
) -> schemas.ChatRead:
    try:
        chat = crud.get_chat(db, str(chat_id))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return schemas.ChatRead.from_orm(chat)


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_chat(
    chat_id: UUID = Path(..., description="Unique chat identifier"),
    db: Session = Depends(get_db),
) -> Response:
    try:
        crud.delete_chat(db, str(chat_id))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{chat_id}/messages", response_model=List[schemas.MessageRead])
def list_messages(
    chat_id: UUID = Path(..., description="Unique chat identifier"),
    db: Session = Depends(get_db),
) -> List[schemas.MessageRead]:
    try:
        messages = crud.list_messages(db, str(chat_id))
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return [schemas.MessageRead.from_orm(message) for message in messages]


@router.post(
    "/{chat_id}/messages",
    response_model=schemas.MessageRead,
    status_code=status.HTTP_201_CREATED,
)
def create_message(
    message_in: schemas.MessageCreate,
    chat_id: UUID = Path(..., description="Unique chat identifier"),
    db: Session = Depends(get_db),
) -> schemas.MessageRead:
    try:
        message = crud.create_message(db, str(chat_id), message_in)
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return schemas.MessageRead.from_orm(message)
