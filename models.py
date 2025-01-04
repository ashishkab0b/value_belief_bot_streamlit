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


class DomainEnum(str, Enum):
    career = "career"
    relationship = "relationship"


class RoleEnum(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class Participant(Base):
    __tablename__ = 'participants'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prolific_id: Mapped[str] = mapped_column(String, nullable=False, unique=False)

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

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    deleted_at: Mapped[datetime] = mapped_column(default=None, nullable=True)

    # Relationships
    participant: Mapped['Participant'] = relationship(back_populates='issues')
    reappraisals: Mapped[list['Reappraisal']] = relationship(back_populates='issue', cascade="all, delete-orphan")


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