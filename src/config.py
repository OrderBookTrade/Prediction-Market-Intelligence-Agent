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


settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
