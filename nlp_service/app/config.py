"""Configuration for the GovPrep-AI NLP/DL service.

Values can be overridden with environment variables prefixed ``NLP_`` or with a
``.env`` file placed in ``nlp_service/`` (see ``.env.example``).
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# nlp_service/ (the folder that contains app/, data/, artifacts/ ...)
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_prefix="NLP_",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    # 8000 is used by the older rag_service, 5000 by the Node backend,
    # 3000 by the React dev server -> 8001 for this service.
    port: int = 8001
    log_level: str = "info"

    data_dir: Path = BASE_DIR / "data"
    artifacts_dir: Path = BASE_DIR / "artifacts"

    # One seed shared by every split and every model so that the
    # TF-IDF / LSTM / BiLSTM / BERT comparison is reproducible.
    random_seed: int = 42


settings = Settings()
