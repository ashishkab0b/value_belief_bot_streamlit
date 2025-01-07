# models.py

from datetime import datetime, timezone
from enum import Enum
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Text,
    ForeignKey,
    DateTime,
    Boolean,
    Enum as SQLAlchemyEnum,
    create_engine,
    Index,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
)
    
class Base(DeclarativeBase):
    pass

class StateEnum(str, Enum):
    start = "start"
    issue = "issue"
    rate_issue = "rate_issue"
    generate_reaps = "generate_reaps"
    summarize_issue = "summarize_issue"
    rate_reap = "rate_reap"
    end = "end"

class DomainEnum(str, Enum):
    career = "career"
    relationship = "relationship"


class RoleEnum(str, Enum):
    user = "user"
    assistant = "assistant"


class Participant(Base):
    __tablename__ = 'participants'
    # id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String, primary_key=True)  # prolific_id
    
    # Data
    cur_state: Mapped[str] = mapped_column(SQLAlchemyEnum(StateEnum), nullable=True, default=StateEnum.start)
    cur_domain: Mapped[str] = mapped_column(SQLAlchemyEnum(DomainEnum), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deleted_at: Mapped[datetime] = mapped_column(default=None, nullable=True)

    # Relationships
    messages: Mapped[list['Message']] = relationship(
        back_populates='participant', cascade="all, delete-orphan"
    )
    issues: Mapped[list['Issue']] = relationship(
        back_populates='participant', cascade="all, delete-orphan"
    )
    reappraisals: Mapped[list['Reappraisal']] = relationship(
        back_populates='participant', cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return (
            f"<Participant(id={self.id}, cur_state={self.cur_state}, "
            f"cur_domain={self.cur_domain}, created_at={self.created_at}, "
            f"updated_at={self.updated_at}, deleted_at={self.deleted_at})>"
        )

class Message(Base):
    __tablename__ = 'messages'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey('participants.id'), nullable=False)

    state: Mapped[str] = mapped_column(String, nullable=False)
    domain: Mapped[str] = mapped_column(SQLAlchemyEnum(DomainEnum), nullable=False)

    # Data
    role: Mapped[str] = mapped_column(SQLAlchemyEnum(RoleEnum), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deleted_at: Mapped[datetime] = mapped_column(default=None, nullable=True)

    # Relationships
    participant: Mapped['Participant'] = relationship(back_populates='messages')
    
    def __repr__(self):
        return (
            f"<Message(id={self.id}, participant_id={self.participant_id}, "
            f"state={self.state}, domain={self.domain}, role={self.role}, "
            f"content={(self.content[:30] + '...') if self.content else 'None'}, "
            f"created_at={self.created_at}, updated_at={self.updated_at}, "
            f"deleted_at={self.deleted_at})>"
        )


class Issue(Base):
    __tablename__ = 'issues'
    
    # Metadata
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    participant_id: Mapped[int] = mapped_column(ForeignKey('participants.id'), nullable=False)
    domain: Mapped[str] = mapped_column(SQLAlchemyEnum(DomainEnum), nullable=True)

    # Data
    neg: Mapped[int] = mapped_column(Integer, nullable=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=True)
    summary: Mapped[str] = mapped_column(Text, nullable=True)
    # interview_complete: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deleted_at: Mapped[datetime] = mapped_column(default=None, nullable=True)

    # Relationships
    participant: Mapped['Participant'] = relationship(back_populates='issues')
    reappraisals: Mapped[list['Reappraisal']] = relationship(back_populates='issue', cascade="all, delete-orphan")
    
    def __repr__(self):
        return (
            f"<Issue(id={self.id}, participant_id={self.participant_id}, "
            f"domain={self.domain}, neg={self.neg}, pos={self.pos}, "
            f"summary={(self.summary[:30] + '...') if self.summary else 'None'}, "
            f"created_at={self.created_at}, updated_at={self.updated_at}, "
            f"deleted_at={self.deleted_at})>"
        )


class Reappraisal(Base):
    __tablename__ = 'reappraisals'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    issue_id: Mapped[int] = mapped_column(ForeignKey('issues.id'), nullable=False)
    participant_id: Mapped[int] = mapped_column(ForeignKey('participants.id'), nullable=False)

    domain: Mapped[str] = mapped_column(SQLAlchemyEnum(DomainEnum), nullable=False)
    reap_num: Mapped[int] = mapped_column(Integer, nullable=False)

    # Data
    text: Mapped[str] = mapped_column(Text, nullable=True)
    success: Mapped[int] = mapped_column(Integer, nullable=True)
    believable: Mapped[int] = mapped_column(Integer, nullable=True)
    valued: Mapped[int] = mapped_column(Integer, nullable=True)
    relevance: Mapped[int] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deleted_at: Mapped[datetime] = mapped_column(default=None, nullable=True)

    # Relationships
    participant: Mapped['Participant'] = relationship(back_populates='reappraisals')
    issue: Mapped['Issue'] = relationship(back_populates='reappraisals')
    
    def __repr__(self):
        return (
            f"<Reappraisal(id={self.id}, issue_id={self.issue_id}, "
            f"participant_id={self.participant_id}, domain={self.domain}, "
            f"reap_num={self.reap_num}, text={(self.text[:30] + '...') if self.text else 'None'}, "
            f"success={self.success}, believable={self.believable}, "
            f"valued={self.valued}, relevance={self.relevance}, "
            f"created_at={self.created_at}, updated_at={self.updated_at}, "
            f"deleted_at={self.deleted_at})>"
        )