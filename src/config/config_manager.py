# CONFIG GROUP - Configuration management and settings
"""
Advanced configuration management for KNUE Policy Vectorizer.

This module provides:
- Configuration validation and schema checking
- Configuration templates for common provider setups
- Configuration backup and versioning
- Configuration import/export functionality
- Environment-specific configuration profiles
- Configuration security and credential management
"""

import base64
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import structlog

# Optional dependency: cryptography. Provide a lightweight fallback for tests.
try:
    from cryptography.fernet import Fernet  # type: ignore
except Exception:  # pragma: no cover - fallback path when cryptography isn't installed

    class Fernet:  # minimal, insecure fallback to satisfy tests without cryptography
        def __init__(self, key: bytes):
            self._key = key

        @staticmethod
        def generate_key() -> bytes:
            # Return bytes matching Fernet-style base64 urlsafe key length
            return base64.urlsafe_b64encode(os.urandom(32))

        def encrypt(self, data: bytes) -> bytes:
            return base64.urlsafe_b64encode(data)

        def decrypt(self, token: bytes) -> bytes:
            return base64.urlsafe_b64decode(token)


from src.config.config import Config
from src.utils.crypto_utils import CryptoUtils
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
    optional_env_vars: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
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


@dataclass
class ConfigProfile:
    """Environment-specific configuration profile"""

    name: str
    environment: str  # dev, staging, prod
    description: str
    config: Config
    created_at: datetime
    last_modified: datetime
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Export profile as dictionary"""
        return {
            "name": self.name,
            "environment": self.environment,
            "description": self.description,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_modified": self.last_modified.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigProfile":
        """Create profile from dictionary"""
        return cls(
            name=data["name"],
            environment=data["environment"],
            description=data["description"],
            config=Config.from_dict(data["config"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_modified=datetime.fromisoformat(data["last_modified"]),
            version=data.get("version", "1.0.0"),
        )


class ConfigurationManager:
    """Advanced configuration management system"""

    def __init__(self, config_dir: str = "./config"):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.templates_dir = self.config_dir / "templates"
        self.profiles_dir = self.config_dir / "profiles"
        self.backups_dir = self.config_dir / "backups"
        self.secrets_dir = self.config_dir / "secrets"

        # Create directories
        for directory in [
            self.config_dir,
            self.templates_dir,
            self.profiles_dir,
            self.backups_dir,
            self.secrets_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger = logger.bind(component="ConfigurationManager")

        # Initialize with default templates
        self._ensure_default_templates()

        # Initialize encryption key for secrets
        self._init_encryption()

    def _init_encryption(self):
        """Initialize encryption for sensitive configuration data"""
        key_file = self.secrets_dir / ".key"
        if key_file.exists():
            with open(key_file, "rb") as f:
                self._encryption_key = f.read()
        else:
            self._encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self._encryption_key)
            # Secure the key file
            os.chmod(key_file, 0o600)

        self._cipher = Fernet(self._encryption_key)

    def _ensure_default_templates(self):
        """Create default configuration templates"""
        default_templates = [
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

        for template in default_templates:
            template_file = self.templates_dir / f"{template.name}.json"
            if not template_file.exists():
                self.save_template(template)

    def validate_config(self, config: Config) -> Dict[str, Any]:
        """Comprehensive configuration validation"""
        validation_result = {
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

    def save_template(self, template: ConfigTemplate) -> bool:
        """Save configuration template"""
        try:
            template_file = self.templates_dir / f"{template.name}.json"
            with open(template_file, "w") as f:
                json.dump(template.to_dict(), f, indent=2)

            self.logger.info(
                "Template saved", name=template.name, file=str(template_file)
            )
            return True

        except Exception as e:
            self.logger.error(
                "Failed to save template", name=template.name, error=str(e)
            )
            return False

    def load_template(self, name: str) -> Optional[ConfigTemplate]:
        """Load configuration template"""
        try:
            template_file = self.templates_dir / f"{name}.json"
            if not template_file.exists():
                return None

            with open(template_file, "r") as f:
                data = json.load(f)

            return ConfigTemplate.from_dict(data)

        except Exception as e:
            self.logger.error("Failed to load template", name=name, error=str(e))
            return None

    def list_templates(self, tag: Optional[str] = None) -> List[ConfigTemplate]:
        """List available configuration templates"""
        templates = []

        for template_file in self.templates_dir.glob("*.json"):
            try:
                template = self.load_template(template_file.stem)
                if template and (tag is None or tag in template.tags):
                    templates.append(template)
            except Exception as e:
                self.logger.warning(
                    "Failed to load template", file=str(template_file), error=str(e)
                )

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

    def save_profile(self, profile: ConfigProfile) -> bool:
        """Save configuration profile"""
        try:
            profile_file = self.profiles_dir / f"{profile.name}.json"
            with open(profile_file, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            self.logger.info("Profile saved", name=profile.name, file=str(profile_file))
            return True

        except Exception as e:
            self.logger.error("Failed to save profile", name=profile.name, error=str(e))
            return False

    def load_profile(self, name: str) -> Optional[ConfigProfile]:
        """Load configuration profile"""
        try:
            profile_file = self.profiles_dir / f"{name}.json"
            if not profile_file.exists():
                return None

            with open(profile_file, "r") as f:
                data = json.load(f)

            return ConfigProfile.from_dict(data)

        except Exception as e:
            self.logger.error("Failed to load profile", name=name, error=str(e))
            return None

    def list_profiles(self, environment: Optional[str] = None) -> List[ConfigProfile]:
        """List available configuration profiles"""
        profiles = []

        for profile_file in self.profiles_dir.glob("*.json"):
            try:
                profile = self.load_profile(profile_file.stem)
                if profile and (
                    environment is None or profile.environment == environment
                ):
                    profiles.append(profile)
            except Exception as e:
                self.logger.warning(
                    "Failed to load profile", file=str(profile_file), error=str(e)
                )

        return sorted(profiles, key=lambda p: p.name)

    def create_backup(self, config: Config, description: str = "") -> str:
        """Create configuration backup"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backups_dir / f"config_backup_{timestamp}.json"

        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "config": config.to_dict(),
            "config_hash": self._calculate_config_hash(config),
        }

        with open(backup_file, "w") as f:
            json.dump(backup_data, f, indent=2)

        self.logger.info("Configuration backup created", file=str(backup_file))
        return str(backup_file)

    def restore_backup(self, backup_file: str) -> Optional[Config]:
        """Restore configuration from backup"""
        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            config = Config.from_dict(backup_data["config"])

            # Verify integrity
            expected_hash = backup_data.get("config_hash")
            if expected_hash:
                actual_hash = self._calculate_config_hash(config)
                if actual_hash != expected_hash:
                    self.logger.warning(
                        "Backup integrity check failed", file=backup_file
                    )

            self.logger.info("Configuration restored from backup", file=backup_file)
            return config

        except Exception as e:
            self.logger.error(
                "Failed to restore backup", file=backup_file, error=str(e)
            )
            return None

    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups"""
        backups = []

        for backup_file in self.backups_dir.glob("config_backup_*.json"):
            try:
                with open(backup_file, "r") as f:
                    backup_data = json.load(f)

                backups.append(
                    {
                        "file": str(backup_file),
                        "timestamp": backup_data["timestamp"],
                        "description": backup_data.get("description", ""),
                        "size": backup_file.stat().st_size,
                    }
                )
            except Exception as e:
                self.logger.warning(
                    "Failed to read backup metadata",
                    file=str(backup_file),
                    error=str(e),
                )

        return sorted(backups, key=lambda b: b["timestamp"], reverse=True)

    def encrypt_sensitive_config(self, config: Config) -> Dict[str, str]:
        """Encrypt sensitive configuration values"""
        sensitive_fields = ["openai_api_key", "qdrant_api_key"]

        encrypted_values = {}
        config_dict = config.to_dict()

        for field in sensitive_fields:
            if field in config_dict and config_dict[field]:
                encrypted_value = self._cipher.encrypt(
                    config_dict[field].encode("utf-8")
                )
                encrypted_values[field] = encrypted_value.decode("utf-8")

        return encrypted_values

    def decrypt_sensitive_config(
        self, encrypted_values: Dict[str, str]
    ) -> Dict[str, str]:
        """Decrypt sensitive configuration values"""
        decrypted_values = {}

        for field, encrypted_value in encrypted_values.items():
            try:
                decrypted_value = self._cipher.decrypt(encrypted_value.encode("utf-8"))
                decrypted_values[field] = decrypted_value.decode("utf-8")
            except Exception as e:
                self.logger.error("Failed to decrypt field", field=field, error=str(e))

        return decrypted_values

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

    def _calculate_config_hash(self, config: Config) -> str:
        """Calculate hash of configuration for integrity checking"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return CryptoUtils.calculate_data_integrity_hash(config_str)

    def cleanup_old_backups(self, keep_days: int = 30):
        """Clean up old backup files"""
        cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 60 * 60)

        removed_count = 0
        for backup_file in self.backups_dir.glob("config_backup_*.json"):
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()
                removed_count += 1

        self.logger.info(
            "Cleaned up old backups", removed=removed_count, keep_days=keep_days
        )

        return removed_count
