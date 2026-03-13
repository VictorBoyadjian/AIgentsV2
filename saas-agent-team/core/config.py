"""
Application configuration using Pydantic Settings.

All configuration is loaded from environment variables and .env files.
No secrets are hardcoded. Every setting has a sensible default where possible.
"""

from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application runtime environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class TaskComplexity(str, Enum):
    """Task complexity levels used for LLM model routing."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    CRITICAL = "critical"


class AgentRole(str, Enum):
    """Available agent roles in the system."""

    ARCHITECT = "architect"
    PM = "pm"
    DEV = "dev"
    QA = "qa"
    SECURITY = "security"
    DEVOPS = "devops"
    RESEARCH = "research"
    FALLBACK = "fallback"


class BatchPriority(str, Enum):
    """Priority levels for batch processing."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class BudgetAlertLevel(str, Enum):
    """Budget alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class BudgetConfig(BaseSettings):
    """Budget and cost control configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    monthly_budget_usd: float = 200.0
    daily_budget_usd: float = 10.0
    warning_threshold: float = 0.70
    critical_threshold: float = 0.90
    emergency_model_override: str = "gpt-4o-mini"
    enable_batch_by_default: bool = True
    cache_ttl_seconds: int = 3600


class LLMConfig(BaseSettings):
    """LLM provider and model configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    # API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Models per agent role
    architect_model: str = "claude-opus-4"
    pm_model: str = "claude-sonnet-4-5"
    dev_model: str = "claude-sonnet-4-5"
    qa_model: str = "claude-haiku-4-5"
    security_model: str = "claude-sonnet-4-5"
    devops_model: str = "claude-haiku-4-5"
    research_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"

    # Feature flags
    enable_prompt_caching: bool = True
    enable_batch_api: bool = True
    enable_cost_tracking: bool = True

    def get_model_for_role(self, role: AgentRole) -> str:
        """Return the configured model name for a given agent role."""
        mapping: dict[AgentRole, str] = {
            AgentRole.ARCHITECT: self.architect_model,
            AgentRole.PM: self.pm_model,
            AgentRole.DEV: self.dev_model,
            AgentRole.QA: self.qa_model,
            AgentRole.SECURITY: self.security_model,
            AgentRole.DEVOPS: self.devops_model,
            AgentRole.RESEARCH: self.research_model,
            AgentRole.FALLBACK: self.fallback_model,
        }
        return mapping.get(role, self.fallback_model)


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://saas_agent:password@localhost:5432/saas_agent_db"
    database_sync_url: str = "postgresql://saas_agent:password@localhost:5432/saas_agent_db"
    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_recycle: int = 3600


class RedisConfig(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"


class WeaviateConfig(BaseSettings):
    """Weaviate vector database configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: str = ""


class ToolsConfig(BaseSettings):
    """External tools configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    github_token: str = ""
    github_org: str = ""
    e2b_api_key: str = ""
    tavily_api_key: str = ""
    firecrawl_api_key: str = ""
    railway_token: str = ""
    railway_project_id: str = ""


class ObservabilityConfig(BaseSettings):
    """Observability and tracing configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    langsmith_api_key: str = ""
    langsmith_project: str = "saas-agent-team"
    langsmith_tracing_v2: bool = True


class APIConfig(BaseSettings):
    """API server configuration."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    api_cors_origins: list[str] = Field(default=["http://localhost:3000"])
    secret_key: str = "change-me-in-production-use-a-real-secret-key"
    environment: Environment = Environment.DEVELOPMENT

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse CORS origins from JSON string or return list as-is."""
        if isinstance(v, str):
            return json.loads(v)  # type: ignore[no-any-return]
        return v  # type: ignore[return-value]


class Settings(BaseSettings):
    """Root settings aggregating all configuration sections."""

    model_config = SettingsConfigDict(env_prefix="", env_file=".env", extra="ignore")

    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    api: APIConfig = Field(default_factory=APIConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
