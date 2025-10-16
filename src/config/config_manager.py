# CONFIG GROUP - Configuration management and settings
"""
Advanced configuration management for KNUE Policy Vectorizer.

This module provides:
- Configuration validation and schema checking
- Configuration templates for common provider setups
- Configuration import/export functionality
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.config.config import Config
from src.utils.providers import EmbeddingProvider, VectorProvider

logger = structlog.get_logger(__name__)


@dataclass
class ConfigTemplate:
    """Configuration template for common setups"""

    name: str
    description: str
    embedding_provider: EmbeddingProvider
    vector_provider: VectorProvider
    config_overrides: Dict[str, Any]
    required_env_vars: List[str]
    optional_env_vars: Optional[List[str]] = None
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.optional_env_vars is None:
            self.optional_env_vars = []
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Export template as dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "embedding_provider": str(self.embedding_provider),
            "vector_provider": str(self.vector_provider),
            "config_overrides": self.config_overrides,
            "required_env_vars": self.required_env_vars,
            "optional_env_vars": self.optional_env_vars,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigTemplate":
        """Create template from dictionary"""
        return cls(
            name=data["name"],
            description=data["description"],
            embedding_provider=EmbeddingProvider(data["embedding_provider"]),
            vector_provider=VectorProvider(data["vector_provider"]),
            config_overrides=data["config_overrides"],
            required_env_vars=data["required_env_vars"],
            optional_env_vars=data.get("optional_env_vars", []),
            tags=data.get("tags", []),
        )


class ConfigurationManager:
    """Advanced configuration management system"""

    # Default templates stored in memory
    _DEFAULT_TEMPLATES = [
        ConfigTemplate(
            name="openai-cloud",
            description="OpenAI embeddings with Qdrant Cloud",
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            config_overrides={
                "openai_model": "text-embedding-3-small",
                "openai_base_url": "https://api.openai.com/v1",
                "vector_size": 1536,
                "max_tokens": 8191,
            },
            required_env_vars=[
                "OPENAI_API_KEY",
                "QDRANT_CLOUD_URL",
                "QDRANT_API_KEY",
            ],
            optional_env_vars=["OPENAI_MODEL"],
            tags=["openai", "cloud", "qdrant", "production"],
        ),
        ConfigTemplate(
            name="production-high-performance",
            description="High-performance production setup with OpenAI and Qdrant Cloud",
            embedding_provider=EmbeddingProvider.OPENAI,
            vector_provider=VectorProvider.QDRANT_CLOUD,
            config_overrides={
                "openai_model": "text-embedding-3-large",
                "openai_base_url": "https://api.openai.com/v1",
                "vector_size": 3072,
                "max_tokens": 8191,
                "max_workers": 8,
                "chunk_threshold": 1000,
            },
            required_env_vars=[
                "OPENAI_API_KEY",
                "QDRANT_CLOUD_URL",
                "QDRANT_API_KEY",
            ],
            optional_env_vars=["MAX_WORKERS"],
            tags=["production", "high-performance", "openai", "cloud"],
        ),
    ]

    def __init__(self) -> None:
        """Initialize configuration manager"""
        self.logger = logger.bind(component="ConfigurationManager")

    def validate_config(self, config: Config) -> Dict[str, Any]:
        """Comprehensive configuration validation"""
        validation_result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
        }

        try:
            # Basic validation
            config.validate()
        except ValueError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))

        # Provider-specific validation
        if config.embedding_provider == EmbeddingProvider.OPENAI:
            if not config.openai_api_key:
                validation_result["errors"].append("OpenAI API key is required")
                validation_result["valid"] = False
            elif len(config.openai_api_key) < 20:
                validation_result["warnings"].append("OpenAI API key seems too short")

            # Model compatibility checks
            if (
                config.openai_model == "text-embedding-3-small"
                and config.vector_size != 1536
            ):
                validation_result["warnings"].append(
                    "text-embedding-3-small produces 1536-dimensional vectors, "
                    f"but vector_size is set to {config.vector_size}"
                )
                validation_result["suggestions"].append("Set vector_size to 1536")

            elif (
                config.openai_model == "text-embedding-3-large"
                and config.vector_size != 3072
            ):
                validation_result["warnings"].append(
                    "text-embedding-3-large produces 3072-dimensional vectors, "
                    f"but vector_size is set to {config.vector_size}"
                )
                validation_result["suggestions"].append("Set vector_size to 3072")

        if config.vector_provider == VectorProvider.QDRANT_CLOUD:
            if not config.qdrant_api_key:
                validation_result["errors"].append("Qdrant Cloud API key is required")
                validation_result["valid"] = False

            if not config.qdrant_cloud_url:
                validation_result["errors"].append("Qdrant Cloud URL is required")
                validation_result["valid"] = False
            elif not config.qdrant_cloud_url.startswith("https://"):
                validation_result["warnings"].append(
                    "Qdrant Cloud URL should use HTTPS"
                )

        # Performance suggestions
        if config.max_workers > 12:
            validation_result["warnings"].append(
                f"max_workers={config.max_workers} might be too high for most systems"
            )

        if config.chunk_threshold > 2000:
            validation_result["warnings"].append(
                "Large chunk_threshold might impact search quality"
            )

        # Security checks
        if config.log_level == "DEBUG":
            validation_result["warnings"].append(
                "DEBUG log level might expose sensitive information in production"
            )

        return validation_result

    def load_template(self, name: str) -> Optional[ConfigTemplate]:
        """Load configuration template from memory"""
        for template in self._DEFAULT_TEMPLATES:
            if template.name == name:
                return template
        return None

    def list_templates(self, tag: Optional[str] = None) -> List[ConfigTemplate]:
        """List available configuration templates"""
        templates = []
        for template in self._DEFAULT_TEMPLATES:
            if tag is None or (template.tags and tag in template.tags):
                templates.append(template)
        return sorted(templates, key=lambda t: t.name)

    def create_config_from_template(
        self, template_name: str, env_overrides: Optional[Dict[str, str]] = None
    ) -> Optional[Config]:
        """Create configuration from template"""
        template = self.load_template(template_name)
        if not template:
            self.logger.error("Template not found", name=template_name)
            return None

        # Check required environment variables
        missing_vars = []
        for var in template.required_env_vars:
            if var not in os.environ and (
                env_overrides is None or var not in env_overrides
            ):
                missing_vars.append(var)

        if missing_vars:
            self.logger.error(
                "Missing required environment variables",
                variables=missing_vars,
                template=template_name,
            )
            return None

        # Start with base configuration
        try:
            config = Config.from_env()
        except Exception:
            config = Config()

        # Apply template overrides
        config_dict = config.to_dict()
        config_dict.update(template.config_overrides)

        # Set provider from template
        config_dict["embedding_provider"] = template.embedding_provider
        config_dict["vector_provider"] = template.vector_provider

        # Apply environment overrides
        if env_overrides:
            config_dict.update(env_overrides)

        # Create final configuration
        final_config = Config.from_dict(config_dict)

        # Validate configuration
        validation = self.validate_config(final_config)
        if not validation["valid"]:
            self.logger.error(
                "Configuration validation failed",
                errors=validation["errors"],
                template=template_name,
            )
            return None

        if validation["warnings"]:
            self.logger.warning(
                "Configuration warnings",
                warnings=validation["warnings"],
                template=template_name,
            )

        return final_config

    def export_config(
        self, config: Config, format: str = "json", include_secrets: bool = False
    ) -> str:
        """
        Export configuration in various formats with security considerations.

        Args:
            config: Configuration to export
            format: Output format ('json', 'env', 'yaml')
            include_secrets: If True, include actual API keys (SECURITY RISK)
                           If False, mask sensitive values for safety

        Returns:
            Formatted configuration string

        Security Note:
            When include_secrets=True, sensitive credentials are exposed in clear text.
            Use with extreme caution and ensure proper file permissions.
        """
        config_dict = config.to_dict()

        if not include_secrets:
            # Mask sensitive values
            sensitive_fields = ["openai_api_key", "qdrant_api_key"]
            for field in sensitive_fields:
                if field in config_dict and config_dict[field]:
                    config_dict[field] = "***MASKED***"

        if format.lower() == "json":
            return json.dumps(config_dict, indent=2)

        elif format.lower() == "env":
            env_lines = []

            # Add security warnings
            if include_secrets:
                env_lines.extend(
                    [
                        "# WARNING: This file contains sensitive credentials in clear text!",
                        "# Do not commit this file to version control or share it publicly.",
                        "# Restrict file permissions: chmod 600 .env",
                        "",
                    ]
                )
            else:
                env_lines.extend(
                    [
                        "# WARNING: Sensitive values are masked for security",
                        "# Use --include-secrets flag with caution to export actual credentials",
                        "",
                    ]
                )

            env_mapping = {
                "embedding_provider": "EMBEDDING_PROVIDER",
                "vector_provider": "VECTOR_PROVIDER",
                "openai_api_key": "OPENAI_API_KEY",
                "openai_model": "OPENAI_MODEL",
                "openai_base_url": "OPENAI_BASE_URL",
                "qdrant_cloud_url": "QDRANT_CLOUD_URL",
                "qdrant_api_key": "QDRANT_API_KEY",
                "qdrant_collection": "COLLECTION_NAME",
                "vector_size": "VECTOR_SIZE",
                "max_tokens": "MAX_TOKEN_LENGTH",
                "log_level": "LOG_LEVEL",
            }

            for key, env_var in env_mapping.items():
                if key in config_dict and config_dict[key]:
                    value = config_dict[key]
                    if isinstance(value, str) and " " in value:
                        value = f'"{value}"'
                    env_lines.append(f"{env_var}={value}")

            return "\n".join(env_lines)

        elif format.lower() == "yaml":
            try:
                import yaml

                return yaml.dump(config_dict, default_flow_style=False, indent=2)
            except ImportError:
                self.logger.error("YAML export requires PyYAML package")
                return ""

        else:
            raise ValueError(f"Unsupported export format: {format}")
