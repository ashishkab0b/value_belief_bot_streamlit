# crud.py

from logger_setup import setup_logger
import os
import uuid
from datetime import datetime, timezone
import random
import yaml
from decimal import Decimal
import logging
from typing import Literal, Optional
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from models import (
    Participant,
    Message,
    Issue,
    Reappraisal,
    DomainEnum,
    StateEnum,
    RoleEnum
)
import streamlit as st
from config import CurrentConfig

logger = setup_logger()

def db_create_message(
    session: Session,
    participant_id: str, 
    role: Literal['user', 'assistant'], 
    content: str, 
    state: Literal[StateEnum.start, StateEnum.issue, StateEnum.rate_issue, StateEnum.generate_reaps, StateEnum.rate_reap, StateEnum.end],
    domain: Literal[DomainEnum.career, DomainEnum.relationship],
    **kwargs) -> Message:
    """
    Create and return a Message record (uncommitted).

    Args:
        session (Session): The database session.
        participant_id (str): The participant ID (ID in participants table -- not prolific ID).
        role (Literal['user', 'assistant']): Indicates whether the message is from the user or the AI assistant.
        content (str): The content of the message.
        state (str): The state of the conversation when the message was sent.
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of the conversation when the message was sent.

    Returns:
        Study: The newly created Study object.
    """
    message = Message(
        participant_id=participant_id,
        role=role,
        content=content,
        state=state,
        domain=domain,
        **kwargs
    )
    logger.debug(f"Creating message: {message}")
    session.add(message)
    return message


def db_create_participant(
    session: Session, 
    prolific_id: str,
    **kwargs
    ) -> Participant:
    """
    Create a new participant record in the database.
    
    Args:
        prolific_id (str): The prolific ID of the participant.
        
    Keyword Args:
        cur_state (str): The current state of the conversation.
        cur_domain (Literal[DomainEnum.career, DomainEnum.relationship]): The current domain of the conversation.
        
    Returns:
        Participant: The newly created Participant object.
    """
    participant = Participant(id=prolific_id, **kwargs)
    logger.debug(f"Creating participant: {participant}")
    session.add(participant)
    return participant

def db_get_participant(
    session: Session,
    id: str
    ) -> Optional[Participant]:
    """
    Retrieve a participant record by prolific ID.
    
    Args:
        id (str): The prolific ID of the participant.
        
    Returns:
        Participant: The Participant object if found, otherwise None.
    """
    stmt = select(Participant).where(Participant.id == id)
    result = session.execute(stmt)
    return result.scalar_one_or_none()


def db_create_issue(
    session: Session,
    participant_id: str,
    domain: Literal[DomainEnum.career, DomainEnum.relationship],
    **kwargs
    ) -> Issue:
    """
    Create and return an Issue record (uncommitted).

    Args:
        session (Session): The database session.
        participant_id (str): The prolific ID of the participant.
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of issue.
        
    Keyword Args:
        neg (int): The negative emotion score before the intervention.
        pos (int): The positive emotion score before the intervention.
        summary (str): A summary of the issue.

    Returns:
        Issue: The newly created Issue object.
    """
    issue = Issue(
        participant_id=participant_id,
        domain=domain,
        **kwargs
    )
    logger.debug(f"Creating issue: {issue}")
    session.add(issue)
    return issue


def db_get_issue_by_id(
    session: Session,
    id: str
    ) -> Optional[Issue]:
    """
    Retrieve an issue record by ID.

    Args:
        session (Session): The database session.
        id (str): The ID of the issue to retrieve.
        
    Returns:
        Issue: The Issue object if found, otherwise None.
    """
    stmt = select(Issue).where(Issue.id == id)
    result = session.execute(stmt)
    return result.scalar_one_or_none()


def db_get_issue_by_participant_and_domain(
    session: Session,
    participant_id: str,
    domain: Literal[DomainEnum.career, DomainEnum.relationship]
) -> Optional[Issue]:
    """
    Retrieve an issue record by participant ID and domain.

    Args:
        session (Session): The database session.
        participant_id (str): The prolific ID of the participant.
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of the issue.
        
    Returns:
        Issue: The Issue object if found, otherwise None.
    """
    stmt = select(Issue).where(
        Issue.participant_id == participant_id,
        Issue.domain == domain
    )
    result = session.execute(stmt)
    return result.scalar_one_or_none()


def db_update_issue(
    session: Session,
    id: str,
    **kwargs
    ) -> Optional[Issue]:
    """
    Update and return an Issue record (uncommitted).

    Args:
        session (Session): The database session.
        id (str): The ID of the issue to update.
        
    Keyword Args:
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of issue.
        neg (int): The negative emotion score before the intervention.
        pos (int): The positive emotion score before the intervention.
        summary (str): A summary of the issue.
        interview_complete (bool): Whether the interview is complete.
        
    Returns:
        Issue: The updated Issue object.
    """
    issue = db_get_issue_by_id(session, id)
    if issue is None:
        return None
    for key, value in kwargs.items():
        setattr(issue, key, value)
    return issue


def db_create_reappraisal(
    session: Session,
    participant_id: str,
    issue_id: int,
    domain: Literal[DomainEnum.career, DomainEnum.relationship],
    reap_num: int,
    **kwargs
    ) -> Reappraisal:
    """
    Create and return a Reappraisal record (uncommitted).
    
    Args:
        session (Session): The database session.
        participant_id (str): The prolific ID of the participant.
        issue_id (int): The issue ID (ID in issues table).
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of the reappraisal.
        reap_num (int): The reappraisal number.
        
    Keyword Args:
        text (str): The text of the reappraisal.
        success (int): The success score of the reappraisal.
        believable (int): The believability score of the reappraisal.
        valued (int): The value score of the reappraisal.
        
    Returns:
        Reappraisal: The newly created Reappraisal object.

    """
    reappraisal = Reappraisal(
        participant_id=participant_id,
        issue_id=issue_id,
        domain=domain,
        reap_num=reap_num,
        **kwargs
    )
    logger.debug(f"Creating reappraisal: {reappraisal}")
    session.add(reappraisal)
    return reappraisal


def db_get_reappraisal_by_id(
    session: Session,
    id: str
    ) -> Optional[Reappraisal]:
    """
    Retrieve a reappraisal record by ID.

    Args:
        session (Session): The database session.
        id (str): The ID of the reappraisal to retrieve.
        
    Returns:
        Reappraisal: The Reappraisal object if found, otherwise None.
    """
    stmt = select(Reappraisal).where(Reappraisal.id == id)
    result = session.execute(stmt)
    return result.scalar_one_or_none()

def db_get_reappraisals_by_participant_and_domain(
    session: Session,
    participant_id: str,
    domain: Literal[DomainEnum.career, DomainEnum.relationship]
) -> list[Reappraisal]:
    """
    Retrieve all reappraisals for a participant and domain ordered by reappraisal number.
    
    Args:
        session (Session): The database session.
        participant_id (str): The participant ID (ID in participants table -- not prolific ID).
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of the reappraisals.
        
    Returns:
        list[Reappraisal]: A list of Reappraisal objects.
    """
    stmt = select(Reappraisal).where(
        Reappraisal.participant_id == participant_id,
        Reappraisal.domain == domain
    ).order_by(Reappraisal.reap_num)
    result = session.execute(stmt)
    return result.scalars().all()
    

def db_update_reappraisal(
    session: Session,
    id: str,
    **kwargs
    ) -> Optional[Reappraisal]:
    """
    Update and return a Reappraisal record (uncommitted).
    
    Args:
        session (Session): The database session.
        id (str): The ID of the reappraisal to update.
        
    Keyword Args:
        domain (Literal[DomainEnum.career, DomainEnum.relationship]): The domain of the reappraisal.
        text (str): The text of the reappraisal.
        success (int): The success score of the reappraisal.
        believable (int): The believability score of the reappraisal.
        valued (int): The value score of the reappraisal.
        
    Returns:
        Reappraisal: The updated Reappraisal object.
    """
    reappraisal = db_get_reappraisal_by_id(session, id)
    if reappraisal is None:
        return None
    for key, value in kwargs.items():
        setattr(reappraisal, key, value)
    return reappraisal

def db_get_messages_by_participant(
    session: Session,
    participant_id: str
) -> list[Message]:
    """
    Retrieve all messages for a participant ordered such that early messages are earlier in list.

    Args:
        session (Session): The database session.
        participant_id (str): The participant ID (ID in participants table -- not prolific ID).
        
    Returns:
        list[Message]: A list of Message objects.
    """
    stmt = select(Message).where(Message.participant_id == participant_id).order_by(Message.created_at)
    result = session.execute(stmt)
    return result.scalars().all()