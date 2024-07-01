from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Cars MML-SG projects"
    API_V1_STR: str = "/api/v1"

    MODEL_PATH: str = "ml_model"

    class Config:
        case_sensitive = True


settings = Settings()
