from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

# 1. Create the Async Database Engine
# This handles the low-level connections to your database (e.g. SQLite, PostgreSQL).
# settings.database_url is pulled from your .env file.
engine = create_async_engine(
    settings.database_url, 
    echo=False  # Set to True if you want to see the raw SQL queries in your console
)

# 2. Create a Session Factory
# This is used to generate 'sessions' (temporary connections) whenever the app needs to talk to the DB.
# expire_on_commit=False is standard for async to prevent errors after saving data.
AsyncSessionLocal = async_sessionmaker(
    engine, 
    expire_on_commit=False
)


class Base(DeclarativeBase):
    """
    The Base class for all Database Models.
    Any class you create (like User or Match) that inherits from this 
    will automatically be mapped to a database table.
    """
    pass


async def get_db() -> AsyncSession:
    """
    FastAPI Dependency: Provides an asynchronous database session.
    It ensures the session is automatically closed after the request is finished.
    Usage in a route: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """
    Initializes the database by creating all defined tables.
    This is called by main.py during the app startup ('lifespan').
    """
    async with engine.begin() as conn:
        # run_sync is required because Base.metadata.create_all is a synchronous SQLAlchemy method
        await conn.run_sync(Base.metadata.create_all)