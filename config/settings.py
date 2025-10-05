# config/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # ---- OpenAI ----
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")

    # ---- App ----
    data_dir: str = Field(default="./data", alias="DATA_DIR")
    timezone: str = Field(default="Asia/Karachi", alias="TIMEZONE")

    # ---- Server/UI ----
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",          # load from .env by default
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

# Singleton accessor
settings = Settings()
