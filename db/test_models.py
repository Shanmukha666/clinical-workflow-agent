from models import Base
from session import engine

# Create all tables
Base.metadata.create_all(bind=engine)

print("✅ Tables created successfully!")