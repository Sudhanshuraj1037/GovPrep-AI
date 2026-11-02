from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"
    GROQ_MAX_TOKENS: int = 1024
    GROQ_TEMPERATURE: float = 0.2
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 32
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    CHROMA_PERSIST_DIR: str = "data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "govprep_knowledge"
    TOP_K_RESULTS: int = 5
    MIN_RELEVANCE_SCORE: float = 0.3
    FAITHFULNESS_THRESHOLD: float = 0.25
    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"

    class Config:
        env_file = ".env"

settings = Settings()
