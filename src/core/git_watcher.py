"""Git repository watcher for detecting changes and managing markdown files."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import InvalidGitRepositoryError, Repo
from git.exc import GitCommandError

from src.utils.logger import setup_logger


class GitWatcher:
    """Watches a Git repository for changes and manages markdown files."""

    def __init__(self, config: Dict[str, str]):
        """Initialize GitWatcher with configuration.

        Args:
            config: Dictionary containing repo_url, branch, and cache_dir
        """
        self.repo_url = config["repo_url"]
        self.branch = config["branch"]
        self.cache_dir = Path(config["cache_dir"])
        self.logger = setup_logger(config.get("log_level", "INFO"), "GitWatcher")

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._repo: Optional[Repo] = None

    @property
    def repo(self) -> Repo:
        """Get the Git repository object, initializing if needed."""
        if self._repo is None:
            if self.cache_dir.exists() and (self.cache_dir / ".git").exists():
                try:
                    self._repo = Repo(str(self.cache_dir))
                    self.logger.info(
                        "Loaded existing repository", path=str(self.cache_dir)
                    )
                except InvalidGitRepositoryError:
                    self.logger.warning(
                        "Invalid git repository, will re-clone",
                        path=str(self.cache_dir),
                    )
                    self._repo = None

            if self._repo is None:
                self._repo = self.clone_repository()

        return self._repo

    def clone_repository(self) -> Repo:
        """Clone the repository to the cache directory.

        Returns:
            Git repository object

        Raises:
            GitCommandError: If cloning fails
        """
        self.logger.info("Cloning repository", url=self.repo_url, branch=self.branch)

        # Clean up existing directory if it exists
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)

        try:
            repo = Repo.clone_from(
                self.repo_url,
                str(self.cache_dir),
                branch=self.branch,
                depth=1,  # Shallow clone for efficiency
            )
            self.logger.info("Repository cloned successfully", path=str(self.cache_dir))
            return repo

        except GitCommandError as e:
            self.logger.error("Failed to clone repository", error=str(e))
            raise

    def pull_updates(self) -> None:
        """Pull latest updates from the remote repository.

        Raises:
            GitCommandError: If pulling fails
        """
        try:
            self.logger.info("Pulling updates from remote")
            origin = self.repo.remotes.origin
            origin.pull()
            self.logger.info("Updates pulled successfully")

        except GitCommandError as e:
            self.logger.error("Failed to pull updates", error=str(e))
            raise

    def get_current_commit(self) -> str:
        """Get the current commit SHA.

        Returns:
            Current commit SHA as string
        """
        commit_sha = self.repo.head.commit.hexsha
        self.logger.debug("Current commit", sha=commit_sha)
        return commit_sha

    def has_changes(self, old_commit: Optional[str], new_commit: str) -> bool:
        """Check if there are changes between commits.

        Args:
            old_commit: Previous commit SHA (None for initial run)
            new_commit: Current commit SHA

        Returns:
            True if there are changes, False otherwise
        """
        if old_commit is None:
            self.logger.info("No previous commit, treating as changed")
            return True

        has_changes = old_commit != new_commit
        self.logger.info(
            "Change detection",
            old_commit=old_commit,
            new_commit=new_commit,
            has_changes=has_changes,
        )
        return has_changes

    def get_markdown_files(self) -> List[str]:
        """Get list of all markdown files in the repository.

        Returns:
            List of relative file paths for markdown files
        """
        # Ensure repository is initialized
        _ = self.repo

        markdown_files = []

        # Use pathlib to find all .md files recursively
        for md_file in self.cache_dir.glob("**/*.md"):
            # Get relative path from cache directory
            relative_path = md_file.relative_to(self.cache_dir)
            relative_path_str = str(relative_path)

            # Exclude README*.md files
            filename = md_file.name
            if filename.upper().startswith("README"):
                self.logger.debug("Excluding README file", file_path=relative_path_str)
                continue

            markdown_files.append(relative_path_str)

        self.logger.info("Found markdown files", count=len(markdown_files))
        self.logger.debug("Markdown files", files=markdown_files)

        return sorted(markdown_files)  # Sort for consistent ordering

    def get_changed_files(
        self, old_commit: str, new_commit: str
    ) -> Tuple[List[str], List[str], List[str], List[Tuple[str, str]]]:
        """Get lists of added, modified, deleted, and renamed markdown files between commits.

        Args:
            old_commit: Previous commit SHA
            new_commit: Current commit SHA

        Returns:
            Tuple of (added_files, modified_files, deleted_files, renamed_files) - all markdown files only
            renamed_files contains tuples of (old_path, new_path)
        """
        try:
            # Handle initial sync when no previous commit is known
            if old_commit is None:
                added_files = [path for path in self.get_markdown_files()]
                self.logger.info(
                    "Initial sync detected",
                    added=len(added_files),
                )
                return added_files, [], [], []

            # Get commit objects
            old_commit_obj = self.repo.commit(old_commit)
            new_commit_obj = self.repo.commit(new_commit)

            # Get diff between commits
            diff = old_commit_obj.diff(new_commit_obj)

            added_files = []
            modified_files = []
            deleted_files = []
            renamed_files = []

            def is_relevant_md(path: Optional[str]) -> bool:
                """Checks if a path is a relevant markdown file."""
                if not path or not path.endswith(".md"):
                    return False
                if Path(path).name.upper().startswith("README"):
                    self.logger.debug(
                        "Excluding README file from changes", file_path=path
                    )
                    return False
                return True

            for item in diff:
                if item.change_type == "R":  # Renamed
                    old_path, new_path = item.a_path, item.b_path
                    if is_relevant_md(old_path) or is_relevant_md(new_path):
                        renamed_files.append((old_path, new_path))
                        self.logger.debug(
                            "Detected rename", old_path=old_path, new_path=new_path
                        )
                    continue

                file_path = item.a_path or item.b_path
                if not is_relevant_md(file_path):
                    continue

                if item.change_type == "A":  # Added
                    added_files.append(file_path)
                elif item.change_type == "M":  # Modified
                    modified_files.append(file_path)
                elif item.change_type == "D":  # Deleted
                    deleted_files.append(file_path)
                elif item.change_type == "T":  # Type changed - treat as modified
                    modified_files.append(file_path)
                    self.logger.debug(
                        "Type change treated as modification", file_path=file_path
                    )

            self.logger.info(
                "File changes detected",
                added=len(added_files),
                modified=len(modified_files),
                deleted=len(deleted_files),
                renamed=len(renamed_files),
            )

            return added_files, modified_files, deleted_files, renamed_files

        except Exception as e:
            self.logger.error(
                "Failed to get changed files",
                old_commit=old_commit,
                new_commit=new_commit,
                error=str(e),
            )
            raise

    def get_file_commit_info(self, file_path: str) -> Dict[str, str]:
        """Get commit information for a specific file.

        Args:
            file_path: Relative path to file in repository

        Returns:
            Dictionary with commit_sha and commit_date
        """
        try:
            # Get the most recent commit that modified this file
            commits = list(self.repo.iter_commits(paths=file_path, max_count=1))

            if not commits:
                # If no commits found, use the current HEAD
                commit = self.repo.head.commit
            else:
                commit = commits[0]

            commit_info = {
                "commit_sha": commit.hexsha,
                "commit_date": commit.committed_datetime.isoformat(),
            }

            self.logger.debug(
                "File commit info",
                file=file_path,
                commit_sha=commit_info["commit_sha"],
                commit_date=commit_info["commit_date"],
            )

            return commit_info

        except Exception as e:
            self.logger.error(
                "Failed to get file commit info", file=file_path, error=str(e)
            )
            raise

    def sync_repository(self) -> Tuple[str, bool]:
        """Sync repository (clone or pull) and check for changes.

        Returns:
            Tuple of (current_commit_sha, has_changes_flag)
        """
        try:
            # Get the repository (clone if needed)
            repo = self.repo

            # Store current commit before pulling
            old_commit = repo.head.commit.hexsha

            # Pull updates (if not freshly cloned)
            if hasattr(self, "_repo") and self._repo is not None:
                self.pull_updates()

            # Get new commit after pull
            new_commit = repo.head.commit.hexsha

            # Check if there are changes
            has_changes = self.has_changes(
                old_commit if old_commit != new_commit else None, new_commit
            )

            return new_commit, has_changes

        except Exception as e:
            self.logger.error("Failed to sync repository", error=str(e))
            raise

    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file in the repository.

        Args:
            file_path: Relative path to file in repository

        Returns:
            File content as string
        """
        full_path = self.cache_dir / file_path

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.logger.debug("Read file content", file=file_path, size=len(content))
            return content

        except Exception as e:
            self.logger.error("Failed to read file", file=file_path, error=str(e))
            raise

    def get_github_file_url(self, file_path: str) -> str:
        """Get GitHub URL for a file.

        Args:
            file_path: Relative path to file in repository

        Returns:
            GitHub URL for the file
        """
        # Convert git URL to GitHub web URL
        if self.repo_url.endswith(".git"):
            base_url = self.repo_url[:-4]  # Remove .git
        else:
            base_url = self.repo_url

        # Construct file URL
        file_url = f"{base_url}/blob/{self.branch}/{file_path}"

        self.logger.debug("Generated GitHub URL", file=file_path, url=file_url)

        return file_url
