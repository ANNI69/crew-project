from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    embeddings_provider: str = "fastembed"  # "fastembed" or "fake" (deterministic, for tests)
    embeddings_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    similarity_threshold: float = 0.35

    data_dir: str = "./data"
    cors_origins: str = "*"

    model_config = {"env_file": ".env", "extra": "ignore"}


def get_settings() -> Settings:
    return Settings()
