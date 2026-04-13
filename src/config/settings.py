from pydantic import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    chroma_db_path: str = "./chroma_db"

    class Config:
        env_file = ".env"

settings = Settings()
