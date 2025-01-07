from sqlalchemy import create_engine
from models import Base
from config import CurrentConfig


POSTGRES_URL = CurrentConfig.ROOT_POSTGRES_URL

def create_database():
    engine = create_engine(POSTGRES_URL, echo=True)
    Base.metadata.create_all(engine)
    print("Database schema created or updated successfully.")

if __name__ == "__main__":
    create_database()
    
    
"""
DROP TABLE IF EXISTS public.issues CASCADE;
DROP TABLE IF EXISTS public.messages CASCADE;
DROP TABLE IF EXISTS public.participants CASCADE;
DROP TABLE IF EXISTS public.reappraisals CASCADE;
DROP TYPE stateenum CASCADE;
DROP TYPE domainenum CASCADE;
DROP TYPE roleenum CASCADE;
"""

"""
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO ashish;
"""