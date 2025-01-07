import os
import toml
from models import DomainEnum, RoleEnum, StateEnum


class BaseConfig:
    
    OPENAI_API_KEY = toml.load(".streamlit/secrets.toml")["OPENAI_API_KEY"]
    POSTGRES_URL = toml.load(".streamlit/secrets.toml")["POSTGRES_URL"]
    ROOT_POSTGRES_URL = toml.load(".streamlit/secrets.toml")["ROOT_POSTGRES_URL"]
    
    LANGCHAIN_TRACING_V2="true"
    LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
    LANGCHAIN_API_KEY=toml.load(".streamlit/secrets.toml")["LANGCHAIN_API_KEY"]
    LANGCHAIN_PROJECT="value_belief_bot_streamlit"
    
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 1.0
    N_REAPPRAISALS = 5
    DOMAINS = [DomainEnum.career, DomainEnum.relationship]
    ERROR_MESSAGE = "I'm sorry, there has been an error. Please contact the researcher on Prolific."


class ProductionConfig(BaseConfig):
    
    DEBUG = False
    LOG_LEVEL = "INFO"


class DevelopmentConfig(BaseConfig):
        
    DEBUG = True
    LOG_LEVEL = "DEBUG"
 
 
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig
}


current_env = os.getenv("ENV_TYPE", "development")
CurrentConfig = config_map.get(current_env, DevelopmentConfig)
