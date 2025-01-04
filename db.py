from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
from models import Base
import toml
from config import CurrentConfig
from contextlib import contextmanager

# Load database URL from configuration
DATABASE_URL = CurrentConfig.POSTGRES_URL

# Create the engine and initialize the database
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(bind=engine)

# Create a configured session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@contextmanager
def get_session():
    """
    Provides a transactional scope for database operations.
    """
    session = SessionLocal()  # Initialize a new database session
    try:
        yield session  # Provide the session to the context
    finally:
        session.close()  # Ensure the session is always closed