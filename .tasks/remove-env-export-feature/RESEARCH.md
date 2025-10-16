---
id: TASK-RESEARCH-REMOVE-ENV-EXPORT
version: 1.0.0
status: active
last-updated: 2025-10-16
owner: team-admin
---

# Research: Remove .env Export Feature

## Objective
Identify and remove all functionality that enables exporting environment configuration to `.env` files for security and policy compliance.

## Current Implementation Analysis

### Export Mechanism
- **CLI Command**: `config-export` (if present in sync_pipeline.py)
- **Function**: `save_config_file()` in config_manager.py
- **Helper**: `_generate_env_content()` (if used for .env generation)
- **Risk**: Exposes sensitive credentials (API keys, URLs) as plaintext files

### Security Concerns
1. **.env files may be committed accidentally** → credentials exposed in git history
2. **Shared file storage** → configuration with secrets could be shared insecurely
3. **Audit trail loss** → no tracking of who/when configs were exported
4. **Policy violation** → repository guidelines discourage direct config file writes

### Current Findings
- Config system supports loading from `.env` files (retained)
- Export capability exists but not heavily documented
- Tests may reference export functions (need cleanup)
- README may mention config-export (need to verify and update)

## Implementation Scope
- **Keep**: `load_config()`, `.env` parsing, environment variable resolution
- **Remove**: 
  - `save_config_file()` function
  - `config-export` CLI command
  - `_generate_env_content()` helper (if unused elsewhere)
  - All export-related tests
  - Documentation references to config export

## Risk Assessment
- **Low Risk**: Export is not core to sync pipeline functionality
- **Dependent Code**: Need to verify no critical paths depend on export
- **Backward Compatibility**: CLI removal may affect existing scripts (acceptable per spec)

## Validation Evidence
- All tests pass after removal
- No import errors or missing function references
- README updated to remove export references
- User-facing docs clarify config-only mode (load, not save)
