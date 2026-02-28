from database import engine, SessionLocal, Base, get_db, init_db

# Initialize database tables
init_db()

# Use get_db for dependencies
def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()