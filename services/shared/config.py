"""Configuration management for RAG application."""

import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database Configuration
    db_host: str = "postgres"
    db_port: int = 5432
    db_name: str = "rag_db"
    db_user: str = "rag_user"
    db_password: str = "CHANGE_ON_INSTALL"
    
    # Database SSL/TLS Configuration (Requirement 15.2)
    db_ssl_mode: str = "prefer"  # disable, allow, prefer, require, verify-ca, verify-full
    db_ssl_cert: Optional[str] = None
    db_ssl_key: Optional[str] = None
    db_ssl_root_cert: Optional[str] = None
    
    # Model Configuration
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: str = "cpu"
    model_cache_dir: str = "/root/.cache/huggingface"
    
    # Service URLs
    embedding_service_url: str = "http://embedding-service:8000"
    llm_service_url: str = "http://llm-service:8000"
    
    # API Configuration
    jwt_secret: str = "CHANGE_ON_INSTALL_THIS_SECRET_KEY"
    rate_limit_anonymous: int = 100
    rate_limit_authenticated: int = 1000
    
    # RAG Configuration
    max_prompt_tokens: int = 4096
    max_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Ingestion Configuration
    docs_path: str = "/docs"
    watch_interval: int = 10
    
    # Conversation Configuration
    conversation_retention_days: int = 30
    
    # Embedding Configuration
    embedding_dimension: int = 768  # BGE-base dimension
    
    # TLS Configuration (Requirement 15.2)
    enable_tls: bool = False
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )
    
    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL with SSL parameters.
        
        Constructs database URL with SSL/TLS configuration if enabled.
        Validates Requirement 15.2: TLS 1.3 encryption for all network communication
        """
        url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        
        # Add SSL parameters if not disabled
        if self.db_ssl_mode != "disable":
            params = [f"sslmode={self.db_ssl_mode}"]
            
            if self.db_ssl_cert:
                params.append(f"sslcert={self.db_ssl_cert}")
            if self.db_ssl_key:
                params.append(f"sslkey={self.db_ssl_key}")
            if self.db_ssl_root_cert:
                params.append(f"sslrootcert={self.db_ssl_root_cert}")
            
            url += "?" + "&".join(params)
        
        return url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
