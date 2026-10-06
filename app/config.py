from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mongodb_uri: str
    mongodb_db: str = "assessment"
    llm_api_key: str
    llm_base_url: str | None = None
    llm_model: str
    allowed_origins: str = "http://localhost:8000"
    llm_timeout_s: float = 20.0
    llm_max_retries: int = 2


settings = Settings()