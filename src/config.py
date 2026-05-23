import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    database_url: str = "sqlite:///./data/markets.db"
    log_level: str = "INFO"
    polymarket_base_url: str = "https://gamma-api.polymarket.com"

    # LLM settings
    claude_model: str = "claude-opus-4-5"
    claude_model_fast: str = "claude-sonnet-4-5"  # cheaper, for extraction tasks

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"  # comma-separated list


settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
