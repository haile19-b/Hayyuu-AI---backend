from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET: str
    JWT_EXPIRE_MINUTES: int
    
    STORAGE_ENDPOINT_URL: str
    STORAGE_ACCESS_KEY_ID: str
    STORAGE_SECRET_ACCESS_KEY: str
    STORAGE_BUCKET_NAME: str
    
    REDIS_URL: str = "redis://localhost:6379/0"
    GEMINI_API_KEY: str | None = None
    MAX_FORM_SIZE_MB: int = 50
    ENABLE_EMBEDDED_WORKER: bool = False
    
    # Neo4j Graph Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "yourpassword123"
    
    APP_NAME: str = "FastAPI Backend"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]
    
    @property
    def clean_postgres_dsn(self) -> str:
        """Returns a libpq-compliant DSN without Prisma-specific params for psycopg and LangGraph."""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        if not parsed.query:
            return self.DATABASE_URL
        query_params = urllib.parse.parse_qs(parsed.query)
        valid_libpq_params = {
            "sslmode", "connect_timeout", "application_name", "sslcert", 
            "sslkey", "sslrootcert", "sslcrl", "sslpassword", "channel_binding"
        }
        filtered = {k: v[0] for k, v in query_params.items() if k in valid_libpq_params}
        new_query = urllib.parse.urlencode(filtered)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    @property
    def prisma_database_url(self) -> str:
        """Returns the database URL formatted for Prisma with optimized pool settings for serverless Neon."""
        import urllib.parse
        parsed = urllib.parse.urlparse(self.DATABASE_URL)
        query_params = urllib.parse.parse_qs(parsed.query) if parsed.query else {}
        
        # Ensure serverless-friendly connection pool parameters for Prisma
        if "pool_timeout" not in query_params:
            query_params["pool_timeout"] = ["30"]
        if "connection_limit" not in query_params:
            query_params["connection_limit"] = ["10"]
            
        flat_params = {k: v[0] for k, v in query_params.items()}
        new_query = urllib.parse.urlencode(flat_params)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()