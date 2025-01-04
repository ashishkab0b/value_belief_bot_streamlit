import csv
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from pathlib import Path
from config import CurrentConfig
from models import Participant, Message, Issue, Reappraisal

# Define the database URL
DATABASE_URL = CurrentConfig.POSTGRES_URL

# Set up the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Specify the directory to save CSV files
output_dir = Path("csv_dumps")
output_dir.mkdir(exist_ok=True)

# Mapping of models
models = [Participant, Message, Issue, Reappraisal]

def dump_models_to_csv(session: Session, models, output_dir: Path):
    """Dumps all rows of each SQLAlchemy model to a CSV file."""
    for model in models:
        table_name = model.__tablename__
        output_file = output_dir / f"{table_name}.csv"
        
        # Query all rows
        rows = session.query(model).all()
        
        if rows:
            # Extract column names
            columns = [column.name for column in model.__table__.columns]
            
            # Write to CSV
            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: getattr(row, column) for column in columns})
            
            print(f"Exported {table_name} to {output_file}")
        else:
            print(f"No data found for {table_name}, skipping.")

# Dump all models
if __name__ == "__main__":
    with Session(engine) as session:
        dump_models_to_csv(session, models, output_dir)