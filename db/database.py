"""
Database Configuration and Session Management for Multi-Agent Clinical System
"""

from __future__ import annotations

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class DatabaseConfig:
    """Database configuration settings"""
    
    # Database URL from environment or default to SQLite for development
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./clinical_agent.db"
    )
    
    # Connection Pool Settings
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour
    
    # Echo SQL queries (for debugging)
    ECHO: bool = os.getenv("DB_ECHO", "False").lower() == "true"
    
    # SSL for production databases
    SSL_MODE: Optional[str] = os.getenv("DB_SSL_MODE", None)
    
    @classmethod
    def get_engine_kwargs(cls) -> dict:
        """Get engine specific kwargs based on database type"""
        kwargs = {
            "pool_size": cls.POOL_SIZE,
            "max_overflow": cls.MAX_OVERFLOW,
            "pool_timeout": cls.POOL_TIMEOUT,
            "pool_recycle": cls.POOL_RECYCLE,
            "pool_pre_ping": True,  # Verify connections before using
            "echo": cls.ECHO,
        }
        
        # SQLite specific settings
        if cls.DATABASE_URL.startswith("sqlite"):
            kwargs.pop("pool_size", None)
            kwargs.pop("max_overflow", None)
            kwargs.pop("pool_timeout", None)
            kwargs.pop("pool_recycle", None)
            kwargs["connect_args"] = {"check_same_thread": False}
        
        # PostgreSQL specific settings
        elif "postgresql" in cls.DATABASE_URL:
            kwargs["poolclass"] = QueuePool
            if cls.SSL_MODE:
                kwargs["connect_args"] = {"sslmode": cls.SSL_MODE}
        
        return kwargs


# ==================== Engine Creation ====================
def init_database():
    """Initialize database - create all tables"""
    try:
        from .models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise
def create_database_engine():
    """Create database engine with proper configuration"""
    
    database_url = DatabaseConfig.DATABASE_URL
    
    # Create engine
    try:
        engine = create_engine(
            database_url,
            **DatabaseConfig.get_engine_kwargs()
        )
        
        # Test connection
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        
        logger.info(f"Database engine created successfully for: {database_url.split('@')[-1] if '@' in database_url else database_url}")
        
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


# Create engine instance
engine = create_database_engine()


# ==================== Session Management ====================

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False  # Keep objects usable after commit
)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI to get database session
    
    Usage:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions
    
    Usage:
        with get_db_context() as db:
            db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database context error: {e}")
        raise
    finally:
        db.close()


# ==================== Connection Pool Monitoring ====================

@event.listens_for(engine, "connect")
def receive_connect(dbapi_connection, connection_record):
    """Log when new connection is created"""
    logger.debug(f"New database connection created: {dbapi_connection}")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log when connection is checked out from pool"""
    logger.debug(f"Database connection checked out from pool")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log when connection is returned to pool"""
    logger.debug(f"Database connection returned to pool")


# ==================== Health Check ====================

def check_database_health() -> dict:
    """
    Check database connection health
    
    Returns:
        dict with health status and details
    """
    try:
        with engine.connect() as conn:
            # Execute simple query
            from sqlalchemy import text
            result = conn.execute(text("SELECT 1")).scalar()
            
            # Get connection pool status
            pool = engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            } if hasattr(pool, "size") else {"type": "sqlite"}
            
            return {
                "status": "healthy" if result == 1 else "degraded",
                "database_url": engine.url.database,
                "pool_status": pool_status,
                "echo_enabled": DatabaseConfig.ECHO
            }
            
    except SQLAlchemyError as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "database_url": engine.url.database
        }
    except Exception as e:
        logger.error(f"Unexpected database health check error: {e}")
        return {
            "status": "unhealthy",
            "error": "Unexpected error during health check"
        }


# ==================== Migration Support ====================

def run_migrations():
    """
    Run database migrations (for development)
    In production, use Alembic for migrations
    """
    try:
        from .models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def drop_all_tables():
    """Drop all tables (for testing only)"""
    if os.getenv("ENVIRONMENT") == "test":
        from .models import Base
        Base.metadata.drop_all(bind=engine)
        logger.warning("All tables dropped - TESTING ONLY")
    else:
        raise RuntimeError("drop_all_tables can only be used in test environment")


# ==================== Session Utilities ====================

def get_session_stats() -> dict:
    """
    Get current session statistics
    
    Returns:
        dict with session pool statistics
    """
    pool = engine.pool
    if hasattr(pool, "size"):
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "total_connections": pool.size() + pool.overflow(),
        }
    return {"type": "sqlite", "note": "SQLite doesn't use connection pooling"}


def close_all_sessions():
    """Close all sessions and dispose engine (for shutdown)"""
    SessionLocal.close_all()
    engine.dispose()
    logger.info("All database sessions closed and engine disposed")


# ==================== Query Helpers ====================

class QueryHelper:
    """Helper class for common database operations"""
    
    @staticmethod
    def paginate(query, page: int = 1, per_page: int = 20):
        """Paginate query results"""
        offset = (page - 1) * per_page
        total = query.count()
        items = query.offset(offset).limit(per_page).all()
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": (total + per_page - 1) // per_page
        }
    
    @staticmethod
    def exists(db: Session, model, **filters) -> bool:
        """Check if record exists"""
        query = db.query(model)
        for key, value in filters.items():
            query = query.filter(getattr(model, key) == value)
        return query.first() is not None
    
    @staticmethod
    def get_or_create(db: Session, model, defaults: dict = None, **kwargs):
        """Get existing record or create new one"""
        instance = db.query(model).filter_by(**kwargs).first()
        if instance:
            return instance, False
        else:
            params = {**kwargs, **(defaults or {})}
            instance = model(**params)
            db.add(instance)
            db.flush()
            return instance, True
    
    @staticmethod
    def bulk_upsert(db: Session, model, records: list, conflict_fields: list):
        """
        Bulk upsert records
        Note: This is a simplified version. For production, use database-specific
        upsert functionality (ON CONFLICT for PostgreSQL)
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        
        if not records:
            return
        
        # For PostgreSQL
        if engine.name == "postgresql":
            stmt = pg_insert(model).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=conflict_fields,
                set_={k: stmt.excluded[k] for k in records[0].keys() if k not in conflict_fields}
            )
            db.execute(stmt)
        else:
            # For SQLite - simple insert or replace
            for record in records:
                db.merge(model(**record))
        
        db.flush()


# ==================== Initialization ====================

def init_db():
    """Initialize database (create tables if not exist)"""
    try:
        run_migrations()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


if __name__ == "__main__":
    # Test database connection
    print("Testing database connection...")
    
    try:
        health = check_database_health()
        print(f"Database health: {health['status']}")
        
        if health['status'] == 'healthy':
            print(f"Database: {health['database_url']}")
            if 'pool_status' in health:
                print(f"Pool status: {health['pool_status']}")
        
        # Initialize tables
        init_db()
        print("Database initialized successfully")
        
    except Exception as e:
        print(f"Database test failed: {e}")