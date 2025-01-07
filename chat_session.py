# chat_session.py

from sqlalchemy.orm import Session
from models import (
    Issue,
    Reappraisal, 
    Message,
    Participant,
    StateEnum,
    DomainEnum,
    RoleEnum,
    )
from db import get_session
from crud import (
    db_get_messages_by_participant,
    db_create_issue,
    db_create_reappraisal
)
from random import random
from config import CurrentConfig

from logger_setup import setup_logger

        
logger = setup_logger()
def load_chat(session: Session, participant: Participant):
    """
    Load a participant's chat session from the database or initialize a new chat session.
    This function must be called within a db session context with an existing participant object.

    Args:
        session (Session): The SQLAlchemy session object.
        participant (Participant): The participant object. Must have attributes `prolific_id`, `cur_domain`, and `cur_state`.
    """
    prolific_id = participant.id
    logger.info(f"Loading chat session for participant.id={prolific_id}")

    # Load current state or set to start
    if getattr(participant, "cur_state", None) is None:
        logger.info(f"Setting initial state to StateEnum.start for participant.id={prolific_id}")
        participant.cur_state = StateEnum.start
    else:
        logger.info(f"Current state for participant.id={prolific_id} is '{participant.cur_state}'")

    # Load current domain or set to random
    if getattr(participant, "cur_domain", None) is None:
        participant.cur_domain = DomainEnum.career if random() < 0.5 else DomainEnum.relationship
        logger.info(f"Setting initial domain to '{participant.cur_domain}' for participant.id={prolific_id}")
    else:
        logger.info(f"Current domain for participant.id={prolific_id} is '{participant.cur_domain}'")
    other_domain = DomainEnum.relationship if participant.cur_domain == DomainEnum.career else DomainEnum.career

    # Load the participant's messages from the database or set to empty list
    logger.info(f"Fetching messages for participant.id={prolific_id}")
    db_messages = db_get_messages_by_participant(session, prolific_id)
    msg_list = [
        {"role": message.role, 
         "content": message.content, 
         "state": message.state, 
         "domain": message.domain}
        for message in db_messages
    ]
    logger.info(f"Loaded {len(msg_list)} messages for participant.id={prolific_id}")

    # Load issue messages from the database
    logger.info(f"Fetching issue messages for participant.id={prolific_id}")
    issue_msg_dict = {
        DomainEnum.career: [
            {"role": msg.role, 
             "content": msg.content, 
             "state": msg.state, 
             "domain": msg.domain}
            for msg in db_messages if msg.state == StateEnum.issue and msg.domain == DomainEnum.career
        ],
        DomainEnum.relationship: [
            {"role": msg.role, 
             "content": msg.content, 
             "state": msg.state, 
             "domain": msg.domain}
            for msg in db_messages if msg.state == StateEnum.issue and msg.domain == DomainEnum.relationship
        ],
    }

    # Load issue ratings
    logger.info(f"Fetching issues for participant.id={prolific_id}")
    db_issues = participant.issues
    career_issue = next((issue for issue in db_issues if issue.domain == DomainEnum.career), None)
    relationship_issue = next((issue for issue in db_issues if issue.domain == DomainEnum.relationship), None)
    issue_dict = {
        DomainEnum.career: {
            "id": getattr(career_issue, 'id', None),
            "neg": getattr(career_issue, 'neg', None),
            "pos": getattr(career_issue, 'pos', None),
            "summary": getattr(career_issue, 'summary', None),
        },
        DomainEnum.relationship: {
            "id": getattr(relationship_issue, 'id', None),
            "neg": getattr(relationship_issue, 'neg', None),
            "pos": getattr(relationship_issue, 'pos', None),
            "summary": getattr(relationship_issue, 'summary', None),
        }
    }

    # If there is no issue, create in database
    if not career_issue:
        logger.info(f"Creating new career issue for participant.id={prolific_id}")
        career_issue = db_create_issue(session, participant_id=prolific_id, domain=DomainEnum.career)
    else:
        logger.info(f"Career issue loaded for participant.id={prolific_id}, id={career_issue.id}")
    if not relationship_issue:
        logger.info(f"Creating new relationship issue for participant.id={prolific_id}")
        relationship_issue = db_create_issue(session, participant_id=prolific_id, domain=DomainEnum.relationship)
    else:
        logger.info(f"Relationship issue loaded for participant.id={prolific_id}, id={relationship_issue.id}")
        

    # Load the reappraisals from the database
    logger.info(f"Fetching reappraisals for participant.id={prolific_id}")
    db_reaps = participant.reappraisals
    reap_dict = {
        DomainEnum.career: [{
            "id": getattr(reap, 'id', None),
            "text": getattr(reap, 'text', None),
            "success": getattr(reap, 'success', None),
            "believable": getattr(reap, 'believable', None),
            "valued": getattr(reap, 'valued', None),
            "relevance": getattr(reap, 'relevance', None),
        }
        for reap in db_reaps if getattr(reap, "domain", None) == DomainEnum.career],
        DomainEnum.relationship: [{
            "id": getattr(reap, 'id', None),
            "text": getattr(reap, 'text', None),
            "success": getattr(reap, 'success', None),
            "believable": getattr(reap, 'believable', None),
            "valued": getattr(reap, 'valued', None),
            "relevance": getattr(reap, 'relevance', None),
        } for reap in db_reaps if getattr(reap, "domain", None) == DomainEnum.relationship]
    }
    logger.info(f"Reappraisals loaded for participant.id={prolific_id}")

    # Create a session state object
    session_state = {
        "llm_model": CurrentConfig.LLM_MODEL,
        "prolific_id": prolific_id,
        "messages": msg_list,
        "issue_messages": issue_msg_dict,
        "reappraisals": reap_dict,
        "issues": issue_dict,
        "cur_domain": participant.cur_domain,
        "other_domain": other_domain,
        "cur_state": participant.cur_state,
        "active_session": True,
    }
    logger.info(f"Session state created for participant.id={prolific_id}: {session_state}")

    return session_state
    