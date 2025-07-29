from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="/chatbot/.env", env_file_encoding="utf-8"
    )

    # Auth
    AUTH_SALT: str
    AUTH_SECRET: str
    DEV_USER_ID: Optional[str] = None

    # PostgresSQL
    POSTGRESQL_URL_NO_SSL: str

    # Redis
    REDIS_URL: str
    REDIS_KEY_PREFIX: str
    REDIS_EXPIRE_TTL: int

    RABBIT_MQ_URL: str

    # Workflow
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    MCP_SERVER_PATH: str
    BRAVE_API_KEY: str
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    LLAMA_BASE_URL: Optional[str] = None
    LLAMA_API_KEY: Optional[str] = None
    DEFAULT_MAX_RUNS_PER_STEP: int = 5

    # LLM
    LLM_API_KEY: str = None
    LLM_BASE_URL: str = None

    # Embedding Configuration
    EMBEDDING_SERVICE_TYPE: str = None
    EMBEDDING_DIMENSION: int = None
    EMBEDDING_BASE_URL: Optional[str] = None
    EMBEDDING_API_KEY: Optional[str] = None
    LOCAL_EMBEDDING_BASE_URL: Optional[str] = None
    LOCAL_EMBEDDING_MODEL_NAME: Optional[str] = None
    LOCAL_EMBEDDING_AUTH_TOKEN: Optional[str] = None
    LOCAL_EMBEDDING_MODEL_PATH: Optional[str] = None

    MAX_CONCURRENCY: str
    MAX_RETRIES: str
    ENVIRONMENT: str
    DOCKER_HOST: str
    DOCKER_SOCKET_PATH: str
    BLOB_READ_WRITE_TOKEN: str
    HOST_PATH: str
    VOYAGE_API_KEY: str
    DOC2X_API_KEY: str

    # s3
    AWS_DEFAULT_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_BUCKET_NAME: str


settings = Settings()
