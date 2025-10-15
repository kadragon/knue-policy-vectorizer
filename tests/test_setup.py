"""Test basic project setup and imports."""

import pytest


def test_imports():
    """Test that all required packages can be imported."""
    # Core dependencies
    import click
    import frontmatter
    import git
    import langchain

    # Utilities
    import pydantic

    # Test dependencies
    import pytest
    import pytest_mock
    import qdrant_client
    import structlog

    # All imports successful
    assert True


def test_python_version():
    """Test Python version compatibility."""
    import sys

    # Should be Python 3.9 or higher
    assert sys.version_info >= (3, 9)


def test_project_structure():
    """Test basic project structure exists."""
    import os

    # Check main directories exist
    assert os.path.exists("src")
    assert os.path.exists("tests")
    assert os.path.exists("config")

    # Check configuration files exist
    assert os.path.exists("pyproject.toml")
    assert os.path.exists("uv.lock")


def test_config_module():
    """Test config module can be imported and used."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from config import Config

    # Test default config creation
    config = Config()
    assert config.repo_url == "https://github.com/kadragon/KNUE-Policy-Hub.git"
    assert config.branch == "main"
    assert config.vector_size == 1536
    assert config.openai_model == "text-embedding-3-large"

    # Test environment-based config
    config_from_env = Config.from_env()
    assert isinstance(config_from_env, Config)


def test_logger_setup():
    """Test logger setup works correctly."""
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    from src.utils.logger import setup_logger

    logger = setup_logger("INFO", "test-logger")
    assert logger is not None

    # Test logging works
    logger.info("Test log message", extra_field="test_value")

    # Should not raise any exception
    assert True
