# Plan: Refactor src/ Folder Structure

## Overview
Execute the refactoring in a feature branch with TDD approach: ensure tests pass at each step.

## Steps
1. **Create new subfolders** in src/: services/, config/, utils/, pipelines/, core/
2. **Move files** to respective folders (create __init__.py in each)
3. **Update imports in src/**: Change relative imports to use new paths
4. **Update imports in tests/**: Update all from src.module to src.subfolder.module
5. **Run tests**: Execute full test suite to verify no breaks
6. **Verify scripts and pipelines**: Run key scripts to ensure functionality
7. **Update documentation**: Check and update any references in README, .agents/, .spec/

## Dependencies
- None (internal refactoring)

## Rollback Plan
- If tests fail: git reset --hard to pre-refactor state
- Backup: Commit after each major step for easy revert

## Risk Mitigation
- Work in feature branch: feat/refactor-src-structure
- Incremental commits: One per subfolder move
- Test after each import update

## Timeline
- Research: Completed
- Spec: Completed
- Plan: Completed
- Implement: ~2-3 hours (depending on import complexity)