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
    GEMINI_MODELS: list[str] = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
    MAX_FORM_SIZE_MB: int = 50
    ENABLE_EMBEDDED_WORKER: bool = False
    
    # Neo4j Graph Database
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "yourpassword123"
    
    APP_NAME: str = "FastAPI Backend"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["*"]
    
    # GitHub App Integration
    GITHUB_APP_ID: str = "5300504"
    GITHUB_APP_SLUG: str = "hayyuu-ai"
    GITHUB_APP_PRIVATE_KEY_PATH: str = "certs/hayyuu-ai.2026-10-13.private-key.pem"
    GITHUB_APP_PRIVATE_KEY: str | None = None

    def get_github_private_key(self) -> str | None:
        """Returns the GitHub App private key either from raw env string or from the .pem file."""
        if self.GITHUB_APP_PRIVATE_KEY and self.GITHUB_APP_PRIVATE_KEY.strip():
            return self.GITHUB_APP_PRIVATE_KEY.replace("\\n", "\n").strip()
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if self.GITHUB_APP_PRIVATE_KEY_PATH:
            path = self.GITHUB_APP_PRIVATE_KEY_PATH
            if not os.path.isabs(path):
                candidate = os.path.join(base_dir, path)
                if os.path.exists(candidate):
                    path = candidate
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return f.read().strip()
        certs_dir = os.path.join(base_dir, "certs")
        if os.path.isdir(certs_dir):
            for cf in os.listdir(certs_dir):
                if cf.endswith(".pem"):
                    with open(os.path.join(certs_dir, cf), "r", encoding="utf-8") as pf:
                        return pf.read().strip()
        return None
    
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