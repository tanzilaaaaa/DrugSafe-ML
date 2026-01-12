"""
Database Configuration
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL', 'sqlite:///drug_interactions.db')
        self.echo = os.environ.get('DATABASE_ECHO', False)
        self.pool_size = int(os.environ.get('DB_POOL_SIZE', 10))
        self.max_overflow = int(os.environ.get('DB_MAX_OVERFLOW', 20))
        self.pool_recycle = int(os.environ.get('DB_POOL_RECYCLE', 3600))
        
    def get_engine(self):
        """Create and return database engine"""
        engine_kwargs = {
            'echo': self.echo,
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow,
            'pool_recycle': self.pool_recycle,
        }
        
        # SQLite doesn't support pool settings
        if 'sqlite' in self.database_url:
            engine_kwargs = {'echo': self.echo}
        
        engine = create_engine(self.database_url, **engine_kwargs)
        return engine
    
    def get_session(self):
        """Create and return database session"""
        engine = self.get_engine()
        Session = scoped_session(sessionmaker(bind=engine))
        return Session()
    
    def init_db(self):
        """Initialize database"""
        engine = self.get_engine()
        # Import models here to avoid circular imports
        # Base.metadata.create_all(engine)
        return engine

# Database configuration instance
db_config = DatabaseConfig()

# Connection pool settings
DB_CONFIG = {
    'url': os.environ.get('DATABASE_URL', 'sqlite:///drug_interactions.db'),
    'echo': os.environ.get('DATABASE_ECHO', False),
    'pool_size': int(os.environ.get('DB_POOL_SIZE', 10)),
    'max_overflow': int(os.environ.get('DB_MAX_OVERFLOW', 20)),
    'pool_recycle': int(os.environ.get('DB_POOL_RECYCLE', 3600)),
}
