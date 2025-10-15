# SPEC-DELTA: Remove .env Export Feature

## Acceptance Criteria
- No .env file generation from CLI
- save_config_file function removed
- config-export command removed
- _generate_env_content function removed if unused
- Tests updated to not call removed functions
- README updated to remove config-export references

## Scope
- Remove CLI commands for .env export
- Keep .env loading functionality (load-config)

## Non-Functional
- No impact on other config loading
- Security improved by not allowing config export