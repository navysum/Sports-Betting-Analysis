from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession  # Async SQLAlchemy engine, session factory, and session type
from sqlalchemy.orm import DeclarativeBase  # Base class for defining ORM models (table mappings)
from app.config import settings  # Application configuration (e.g., database URL, env variables)

engine = create_async_engine(settings.database_url, echo=False)  # Create async database engine (connection to DB)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)  # Factory for creating async DB sessions


class Base(DeclarativeBase):  # Base class for all ORM models (table definitions)
    pass


async def get_db() -> AsyncSession:  # Dependency that provides a DB session per request
    async with AsyncSessionLocal() as session:  # Create a new async DB session
        yield session  # Give session to request and auto-close after use


async def init_db():  # Initialize database (create tables)
    async with engine.begin() as conn:  # Open async connection/transaction
        await conn.run_sync(Base.metadata.create_all)  # Create all tables defined in models