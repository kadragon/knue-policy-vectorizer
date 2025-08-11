"""Test Git repository watcher functionality (TDD approach)."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from git import Repo
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestGitWatcher:
    """Test Git repository watcher functionality."""
    
    def test_git_watcher_initialization(self):
        """Test GitWatcher can be initialized with proper configuration."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        assert watcher.repo_url == config['repo_url']
        assert watcher.branch == config['branch']
        assert str(watcher.cache_dir) == config['cache_dir']
    
    def test_git_watcher_clone_repository(self):
        """Test that GitWatcher can clone a repository."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main', 
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        # Mock the git operations
        with patch('git.Repo.clone_from') as mock_clone:
            mock_repo = Mock()
            mock_clone.return_value = mock_repo
            
            result = watcher.clone_repository()
            
            mock_clone.assert_called_once_with(
                config['repo_url'],
                str(watcher.cache_dir),
                branch=config['branch'],
                depth=1
            )
            assert result == mock_repo
    
    def test_git_watcher_pull_updates(self):
        """Test that GitWatcher can pull updates from remote."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        # Mock existing repository
        mock_repo = Mock()
        mock_origin = Mock()
        mock_repo.remotes.origin = mock_origin
        watcher._repo = mock_repo
        
        watcher.pull_updates()
        
        mock_origin.pull.assert_called_once()
    
    def test_git_watcher_get_current_commit(self):
        """Test getting current commit SHA."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        test_commit = "abc123def456"
        
        mock_repo = Mock()
        mock_repo.head.commit.hexsha = test_commit
        watcher._repo = mock_repo
        
        commit = watcher.get_current_commit()
        assert commit == test_commit
    
    def test_git_watcher_has_changes(self):
        """Test detecting if repository has changes."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        # Test case 1: No previous commit (first run)
        assert watcher.has_changes(None, "new_commit") == True
        
        # Test case 2: Same commit
        assert watcher.has_changes("same_commit", "same_commit") == False
        
        # Test case 3: Different commits
        assert watcher.has_changes("old_commit", "new_commit") == True
    
    def test_git_watcher_get_markdown_files(self):
        """Test getting list of markdown files from repository."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        # Mock file system using pathlib.Path.glob
        with patch('pathlib.Path.glob') as mock_glob:
            mock_files = [
                Path('/tmp/test_cache/policy1.md'),
                Path('/tmp/test_cache/subdir/policy2.md'),
                Path('/tmp/test_cache/README.md')
            ]
            mock_glob.return_value = mock_files
            
            files = watcher.get_markdown_files()
            
            # Should return relative paths (sorted)
            expected_files = [
                'README.md',
                'policy1.md',
                'subdir/policy2.md'
            ]
            assert files == expected_files
    
    def test_git_watcher_get_changed_files(self):
        """Test getting list of changed files between commits."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        mock_repo = Mock()
        mock_commit_old = Mock()
        mock_commit_new = Mock()
        
        # Mock git diff
        mock_diff_items = [
            Mock(a_path='policy1.md', change_type='M'),  # Modified
            Mock(a_path='policy2.md', change_type='A'),  # Added
            Mock(a_path='policy3.md', change_type='D'),  # Deleted
            Mock(a_path='non_md_file.txt', change_type='M')  # Non-markdown
        ]
        
        mock_repo.commit.side_effect = [mock_commit_old, mock_commit_new]
        mock_commit_old.diff.return_value = mock_diff_items
        watcher._repo = mock_repo
        
        added, modified, deleted = watcher.get_changed_files('old_sha', 'new_sha')
        
        assert 'policy2.md' in added
        assert 'policy1.md' in modified
        assert 'policy3.md' in deleted
        assert 'non_md_file.txt' not in (added + modified + deleted)
    
    def test_git_watcher_get_file_commit_info(self):
        """Test getting commit information for a specific file."""
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/test/test-repo.git',
            'branch': 'main',
            'cache_dir': '/tmp/test_cache'
        }
        
        watcher = GitWatcher(config)
        
        mock_repo = Mock()
        mock_commit = Mock()
        mock_commit.committed_datetime.isoformat.return_value = '2024-01-15T10:30:00'
        mock_commit.hexsha = 'abc123'
        
        mock_repo.iter_commits.return_value = [mock_commit]
        watcher._repo = mock_repo
        
        commit_info = watcher.get_file_commit_info('policy1.md')
        
        assert commit_info['commit_sha'] == 'abc123'
        assert commit_info['commit_date'] == '2024-01-15T10:30:00'
        mock_repo.iter_commits.assert_called_once_with(paths='policy1.md', max_count=1)


class TestGitWatcherIntegration:
    """Integration tests for GitWatcher (will require actual git operations)."""
    
    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary git repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo = Repo.init(temp_dir)
        
        # Create initial commit
        test_file = os.path.join(temp_dir, 'test.md')
        with open(test_file, 'w') as f:
            f.write('# Test Document\n\nThis is a test.')
        
        repo.index.add(['test.md'])
        repo.index.commit('Initial commit')
        
        yield temp_dir, repo
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_git_watcher_real_operations(self, temp_git_repo):
        """Test GitWatcher with real git operations."""
        temp_dir, repo = temp_git_repo
        
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': temp_dir,  # Use local path for testing
            'branch': 'master',  # Default branch for new repos is usually master
            'cache_dir': temp_dir
        }
        
        watcher = GitWatcher(config)
        watcher._repo = repo  # Use the existing repo directly
        
        # Test getting markdown files
        files = watcher.get_markdown_files()
        assert 'test.md' in files
        
        # Test getting current commit
        current_commit = watcher.get_current_commit()
        assert len(current_commit) == 40  # SHA-1 hash length
    
    @pytest.mark.integration
    def test_git_watcher_with_real_repository(self):
        """Integration test with actual KNUE Policy Hub repository."""
        # This test will be skipped in CI/CD but can be run manually
        pytest.skip("Manual integration test - requires network access")
        
        from git_watcher import GitWatcher
        
        config = {
            'repo_url': 'https://github.com/kadragon/KNUE-Policy-Hub.git',
            'branch': 'main',
            'cache_dir': tempfile.mkdtemp()
        }
        
        try:
            watcher = GitWatcher(config)
            
            # Test cloning
            repo = watcher.clone_repository()
            assert repo is not None
            
            # Test getting markdown files
            files = watcher.get_markdown_files()
            assert len(files) > 0
            assert all(f.endswith('.md') for f in files)
            
            # Test getting commit info
            current_commit = watcher.get_current_commit()
            assert current_commit is not None
            
        finally:
            # Cleanup
            if os.path.exists(config['cache_dir']):
                shutil.rmtree(config['cache_dir'])


# Tests to verify our test setup is correct
def test_git_module_available():
    """Test that GitPython is available."""
    import git
    assert hasattr(git, 'Repo')


def test_pathlib_available():
    """Test that pathlib is available for file operations."""
    from pathlib import Path
    test_path = Path('/tmp')
    assert test_path.exists() or not test_path.exists()  # Either is fine


def test_tempfile_operations():
    """Test that we can create and cleanup temporary directories."""
    temp_dir = tempfile.mkdtemp()
    assert os.path.exists(temp_dir)
    
    # Create a test file
    test_file = os.path.join(temp_dir, 'test.txt')
    with open(test_file, 'w') as f:
        f.write('test content')
    
    assert os.path.exists(test_file)
    
    # Cleanup
    shutil.rmtree(temp_dir)
    assert not os.path.exists(temp_dir)