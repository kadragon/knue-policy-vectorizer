#!/usr/bin/env python3
"""Integration test script for GitWatcher with real KNUE Policy Hub repository."""

import os
import shutil
import sys
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from git_watcher import GitWatcher
from logger import setup_logger


def main():
    """Test GitWatcher with the real KNUE Policy Hub repository."""
    logger = setup_logger("INFO", "GitWatcher-Test")

    print("🧪 Testing GitWatcher with real KNUE Policy Hub repository")

    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Using temporary directory: {temp_dir}")

    try:
        # Configure GitWatcher
        config = {
            "repo_url": "https://github.com/kadragon/KNUE-Policy-Hub.git",
            "branch": "main",
            "cache_dir": temp_dir,
        }

        print(f"⚙️  Configuration:")
        print(f"   - Repository: {config['repo_url']}")
        print(f"   - Branch: {config['branch']}")
        print(f"   - Cache directory: {config['cache_dir']}")

        # Initialize GitWatcher
        print("\n🔧 Initializing GitWatcher...")
        watcher = GitWatcher(config)

        # Test cloning
        print("📥 Cloning repository...")
        repo = watcher.clone_repository()
        print(f"✅ Repository cloned successfully")

        # Get current commit
        print("\n🔍 Getting current commit...")
        current_commit = watcher.get_current_commit()
        print(f"📌 Current commit: {current_commit}")

        # Get markdown files
        print("\n📄 Getting markdown files...")
        md_files = watcher.get_markdown_files()
        print(f"📊 Found {len(md_files)} markdown files:")

        # Display first few files
        for i, file in enumerate(md_files[:10]):
            print(f"   {i+1}. {file}")

        if len(md_files) > 10:
            print(f"   ... and {len(md_files) - 10} more files")

        # Test file commit info
        if md_files:
            test_file = md_files[0]
            print(f"\n📋 Getting commit info for: {test_file}")
            commit_info = watcher.get_file_commit_info(test_file)
            print(f"   - Commit SHA: {commit_info['commit_sha']}")
            print(f"   - Commit date: {commit_info['commit_date']}")

            # Test file content
            print(f"\n📖 Reading file content...")
            content = watcher.get_file_content(test_file)
            print(f"   - Content length: {len(content)} characters")
            print(f"   - First 100 characters: {content[:100]}...")

            # Test GitHub URL generation
            github_url = watcher.get_github_file_url(test_file)
            print(f"   - GitHub URL: {github_url}")

        # Test sync functionality
        print(f"\n🔄 Testing sync functionality...")
        new_commit, has_changes = watcher.sync_repository()
        print(f"   - New commit: {new_commit}")
        print(f"   - Has changes: {has_changes}")

        print(f"\n🎉 All GitWatcher integration tests passed!")
        print(
            f"💡 GitWatcher is working correctly with the KNUE Policy Hub repository!"
        )

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        logger.error("Integration test failed", error=str(e))
        sys.exit(1)

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            print(f"\n🗑️  Cleaning up temporary directory...")
            shutil.rmtree(temp_dir)
            print(f"✅ Cleanup completed")


if __name__ == "__main__":
    main()
