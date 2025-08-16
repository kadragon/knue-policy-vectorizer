"""Test Git repository watcher functionality (TDD approach)."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from git import Repo


@pytest.fixture(autouse=True)
def sys_path_src():
    """Temporarily add src to sys.path for imports."""
    original = list(sys.path)
    src_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        yield
    finally:
        sys.path[:] = original


@pytest.fixture
def watcher_config(tmp_path):
    """Provide a clean, per-test configuration using tmp_path."""
    return {
        "repo_url": "https://github.com/test/test-repo.git",
        "branch": "main",
        "cache_dir": str(tmp_path / "cache"),
    }


class TestGitWatcher:
    """Test Git repository watcher functionality."""

    def test_git_watcher_initialization(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)
        assert watcher.repo_url == watcher_config["repo_url"]
        assert watcher.branch == watcher_config["branch"]
        assert str(watcher.cache_dir) == watcher_config["cache_dir"]

    def test_git_watcher_clone_repository(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        # Ensure cache dir exists (GitWatcher may rely on it)
        os.makedirs(watcher.cache_dir, exist_ok=True)

        with patch("git.Repo.clone_from") as mock_clone:
            mock_repo = Mock()
            mock_clone.return_value = mock_repo

            result = watcher.clone_repository()

            mock_clone.assert_called_once_with(
                watcher_config["repo_url"],
                str(watcher.cache_dir),
                branch=watcher_config["branch"],
                depth=1,
            )
            assert result == mock_repo

    def test_git_watcher_pull_updates(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_origin = Mock()
        mock_repo.remotes.origin = mock_origin
        watcher._repo = mock_repo

        watcher.pull_updates()

        mock_origin.pull.assert_called_once()

    def test_git_watcher_get_current_commit(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        test_commit = "abc123def456"
        mock_repo = Mock()
        mock_repo.head.commit.hexsha = test_commit
        watcher._repo = mock_repo

        commit = watcher.get_current_commit()
        assert commit == test_commit

    def test_git_watcher_has_changes(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        # Test case 1: No previous commit (first run)
        assert watcher.has_changes(None, "new_commit") == True

        # Test case 2: Same commit
        assert watcher.has_changes("same_commit", "same_commit") == False

        # Test case 3: Different commits
        assert watcher.has_changes("old_commit", "new_commit") == True

    def test_git_watcher_get_markdown_files(self, watcher_config, tmp_path):
        from unittest.mock import Mock

        from git_watcher import GitWatcher

        # Prepare a realistic repo cache directory structure
        cache_dir = Path(watcher_config["cache_dir"])
        (cache_dir / "subdir").mkdir(parents=True, exist_ok=True)

        # Create files
        (cache_dir / "policy1.md").write_text("# Policy 1")
        (cache_dir / "subdir" / "policy2.md").write_text("# Policy 2")
        (cache_dir / "README.md").write_text("# Readme")
        (cache_dir / "notes.txt").write_text("not markdown")

        watcher = GitWatcher(watcher_config)
        # Mock the repo property to avoid git operations
        watcher._repo = Mock()
        files = watcher.get_markdown_files()

        expected_files = ["policy1.md", "subdir/policy2.md"]
        assert files == expected_files

    def test_git_watcher_get_changed_files(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_commit_old = Mock()
        mock_commit_new = Mock()

        mock_diff_items = [
            Mock(a_path="policy1.md", change_type="M"),
            Mock(a_path="policy2.md", change_type="A"),
            Mock(a_path="policy3.md", change_type="D"),
            Mock(a_path="non_md_file.txt", change_type="M"),
        ]

        mock_repo.commit.side_effect = [mock_commit_old, mock_commit_new]
        mock_commit_old.diff.return_value = mock_diff_items
        watcher._repo = mock_repo

        added, modified, deleted, renamed = watcher.get_changed_files(
            "old_sha", "new_sha"
        )

        assert added == ["policy2.md"]
        assert modified == ["policy1.md"]
        assert deleted == ["policy3.md"]
        assert renamed == []

    def test_git_watcher_get_file_commit_info(self, watcher_config):
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_commit = Mock()
        mock_commit.committed_datetime.isoformat.return_value = "2024-01-15T10:30:00"
        mock_commit.hexsha = "abc123"

        mock_repo.iter_commits.return_value = [mock_commit]
        watcher._repo = mock_repo

        commit_info = watcher.get_file_commit_info("policy1.md")

        assert commit_info["commit_sha"] == "abc123"
        assert commit_info["commit_date"] == "2024-01-15T10:30:00"
        mock_repo.iter_commits.assert_called_once_with(paths="policy1.md", max_count=1)

    def test_git_watcher_get_changed_files_with_renames(self, watcher_config):
        """Test handling of renamed files."""
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_commit_old = Mock()
        mock_commit_new = Mock()

        mock_diff_items = [
            Mock(a_path="old_policy.md", b_path="new_policy.md", change_type="R"),
            Mock(a_path="policy1.md", b_path="policy1.md", change_type="M"),
            Mock(
                a_path="old_file.txt", b_path="new_file.txt", change_type="R"
            ),  # Non-markdown rename
            Mock(
                a_path="README.md", b_path="README_NEW.md", change_type="R"
            ),  # README rename (should be ignored)
        ]

        mock_repo.commit.side_effect = [mock_commit_old, mock_commit_new]
        mock_commit_old.diff.return_value = mock_diff_items
        watcher._repo = mock_repo

        added, modified, deleted, renamed = watcher.get_changed_files(
            "old_sha", "new_sha"
        )

        assert added == []
        assert modified == ["policy1.md"]
        assert deleted == []
        assert renamed == [("old_policy.md", "new_policy.md")]

    def test_git_watcher_get_changed_files_with_type_changes(self, watcher_config):
        """Test handling of type changed files."""
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_commit_old = Mock()
        mock_commit_new = Mock()

        mock_diff_items = [
            Mock(a_path="policy1.md", change_type="T"),
            Mock(a_path="policy2.md", change_type="M"),
            Mock(a_path="script.sh", change_type="T"),  # Non-markdown type change
        ]

        mock_repo.commit.side_effect = [mock_commit_old, mock_commit_new]
        mock_commit_old.diff.return_value = mock_diff_items
        watcher._repo = mock_repo

        added, modified, deleted, renamed = watcher.get_changed_files(
            "old_sha", "new_sha"
        )

        assert added == []
        assert modified == [
            "policy1.md",
            "policy2.md",
        ]  # Type change treated as modification
        assert deleted == []
        assert renamed == []

    def test_git_watcher_get_changed_files_rename_edge_cases(self, watcher_config):
        """Test edge cases for rename handling."""
        from git_watcher import GitWatcher

        watcher = GitWatcher(watcher_config)

        mock_repo = Mock()
        mock_commit_old = Mock()
        mock_commit_new = Mock()

        mock_diff_items = [
            # Rename from .md to non-.md (should track old path)
            Mock(a_path="policy.md", b_path="policy.txt", change_type="R"),
            # Rename from non-.md to .md (should track new path)
            Mock(a_path="document.txt", b_path="document.md", change_type="R"),
            # Both paths are .md
            Mock(a_path="old.md", b_path="new.md", change_type="R"),
        ]

        mock_repo.commit.side_effect = [mock_commit_old, mock_commit_new]
        mock_commit_old.diff.return_value = mock_diff_items
        watcher._repo = mock_repo

        added, modified, deleted, renamed = watcher.get_changed_files(
            "old_sha", "new_sha"
        )

        assert added == []
        assert modified == []
        assert deleted == []
        assert renamed == [
            ("policy.md", "policy.txt"),  # Old path is .md
            ("document.txt", "document.md"),  # New path is .md
            ("old.md", "new.md"),  # Both paths are .md
        ]


class TestGitWatcherIntegration:
    """Integration tests for GitWatcher (will require actual git operations)."""

    @pytest.fixture
    def temp_git_repo(self, tmp_path):
        """Create a temporary git repository for testing."""
        temp_dir = tmp_path / "repo"
        temp_dir.mkdir(parents=True, exist_ok=True)
        repo = Repo.init(str(temp_dir))

        # Create initial commit
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test Document\n\nThis is a test.")
        repo.index.add(["test.md"])
        repo.index.commit("Initial commit")

        yield str(temp_dir), repo

    def test_git_watcher_real_operations(self, temp_git_repo):
        temp_dir, repo = temp_git_repo

        from git_watcher import GitWatcher

        branch_name = repo.active_branch.name

        config = {"repo_url": temp_dir, "branch": branch_name, "cache_dir": temp_dir}

        watcher = GitWatcher(config)
        watcher._repo = repo

        files = watcher.get_markdown_files()
        assert "test.md" in files

        current_commit = watcher.get_current_commit()
        assert isinstance(current_commit, str) and len(current_commit) in (40, 64)

    @pytest.mark.integration
    def test_git_watcher_with_real_repository(self, tmp_path):
        """Integration test with actual KNUE Policy Hub repository."""
        # This test will be skipped in CI/CD but can be run manually
        pytest.skip("Manual integration test - requires network access")

        from git_watcher import GitWatcher

        config = {
            "repo_url": "https://github.com/kadragon/KNUE-Policy-Hub.git",
            "branch": "main",
            "cache_dir": str(tmp_path / "policy_hub_cache"),
        }

        watcher = GitWatcher(config)

        # Test cloning
        repo = watcher.clone_repository()
        assert repo is not None

        # Test getting markdown files
        files = watcher.get_markdown_files()
        assert len(files) > 0
        assert all(f.endswith(".md") for f in files)

        # Test getting commit info
        current_commit = watcher.get_current_commit()
        assert current_commit is not None


# Tests to verify our test setup is correct
def test_git_module_available():
    """Test that GitPython is available."""
    import git

    assert hasattr(git, "Repo")


def test_pathlib_available():
    """Test that pathlib is available for file operations."""
    from pathlib import Path

    test_path = Path("/tmp")
    assert test_path.exists() or not test_path.exists()  # Either is fine


def test_tempfile_operations():
    """Test that we can create and cleanup temporary directories."""
    temp_dir = tempfile.mkdtemp()
    assert os.path.exists(temp_dir)

    # Create a test file
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test content")

    assert os.path.exists(test_file)

    # Cleanup
    shutil.rmtree(temp_dir)
    assert not os.path.exists(temp_dir)


class TestGitWatcherDetails:
    """Additional tests for URL building, discovery rules, and sync behavior."""

    def _make_watcher(self, tmp_path, repo_url="https://example.com/repo.git", branch="main"):
        from git_watcher import GitWatcher

        cache_dir = tmp_path / "cache"
        config = {
            "repo_url": repo_url,
            "branch": branch,
            "cache_dir": str(cache_dir),
            "log_level": "INFO",
        }
        watcher = GitWatcher(config)
        watcher._repo = Mock()  # Prevent real git operations
        return watcher, cache_dir

    @pytest.mark.parametrize(
        "repo_url,branch,file_path,expected",
        [
            (
                "https://github.com/test/repo.git",
                "main",
                "dir/file.md",
                "https://github.com/test/repo/blob/main/dir/file.md",
            ),
            (
                "https://github.com/test/repo",
                "develop",
                "한글/규정.md",
                "https://github.com/test/repo/blob/develop/한글/규정.md",
            ),
        ],
    )
    def test_get_github_file_url(self, tmp_path, repo_url, branch, file_path, expected):
        watcher, _ = self._make_watcher(tmp_path, repo_url=repo_url, branch=branch)
        assert watcher.get_github_file_url(file_path) == expected

    def test_markdown_discovery_excludes_readme_nested(self, tmp_path):
        watcher, cache_dir = self._make_watcher(tmp_path)

        # Create nested structure with README files and markdowns
        (cache_dir / "sub").mkdir(parents=True, exist_ok=True)
        (cache_dir / "README.md").write_text("# Top Readme")
        (cache_dir / "sub" / "README.md").write_text("# Nested Readme")
        (cache_dir / "policy1.md").write_text("# Policy 1")
        (cache_dir / "sub" / "policy2.md").write_text("# Policy 2")
        (cache_dir / "notes.txt").write_text("not markdown")

        files = watcher.get_markdown_files()
        assert files == ["policy1.md", "sub/policy2.md"]

    def test_get_file_content_missing_raises(self, tmp_path):
        watcher, _ = self._make_watcher(tmp_path)
        with pytest.raises(FileNotFoundError):
            watcher.get_file_content("does/not/exist.md")

    def test_sync_repository_change_detection(self, tmp_path):
        watcher, _ = self._make_watcher(tmp_path)

        mock_repo = watcher._repo
        mock_head = Mock()
        mock_commit = Mock()
        mock_commit.hexsha = "old_sha"
        mock_head.commit = mock_commit
        mock_repo.head = mock_head

        def _pull_side_effect():
            # After pull, HEAD should point to a new commit
            mock_commit.hexsha = "new_sha"

        with patch.object(watcher, "pull_updates", side_effect=_pull_side_effect) as mock_pull:
            new_commit, has_changes = watcher.sync_repository()

        assert mock_pull.called is True
        assert new_commit == "new_sha"
        assert has_changes is True
