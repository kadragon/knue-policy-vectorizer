# KNUE Policy Vectorizer

한국교원대학교 정책 문서를 OpenAI 임베딩과 Qdrant Cloud에 동기화하는 자동화 파이프라인입니다.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Vector DB](https://img.shields.io/badge/vector%20db-Qdrant%20Cloud-orange)](https://qdrant.tech)
[![Embeddings](https://img.shields.io/badge/embeddings-OpenAI%20text--embedding--3-purple)](https://platform.openai.com/docs)
[![Workflow](https://img.shields.io/badge/automation-GitHub%20Actions-black)](.github/workflows/daily-r2-sync.yml)
[![TDD](https://img.shields.io/badge/development-TDD-green)](./tests)

## 📋 개요

KNUE Policy Vectorizer는 [한국교원대학교 정책 Hub](https://github.com/kadragon/KNUE-Policy-Hub) 저장소의 마크다운 정책 문서를 자동으로 수집·전처리하여 OpenAI 임베딩(1536차원)으로 변환하고, Qdrant Cloud에 저장하는 파이프라인입니다. Cloudflare R2를 통해 정제된 원본 문서를 보관하며, GitHub Actions로 일정 기반 동기화를 수행합니다.

### 🎯 주요 기능

- **Git 변경 감시**: 정책 저장소의 커밋 변화를 추적하고 변경된 문서만 증분 처리
- **문서 전처리**: front matter 제거, 제목 추출, 스마트 청킹(800/200 토큰), 메타데이터 생성
- **OpenAI 임베딩**: `text-embedding-3-small`/`large` 모델 지원, 배치 처리 및 토큰 검증
- **Qdrant Cloud 저장소**: HTTPS + API Key 인증, 컬렉션 자동 생성/검증, 헬스 체크
- **Cloudflare R2 백업**: 최근 문서 스냅샷과 소프트 삭제 영역 관리
- **CLI 유틸리티**: `sync`, `reindex`, `health`, `configure`, `test-providers`, `migrate` 명령 제공
- **마이그레이션 도구**: Qdrant Cloud 클러스터 간 백업/복원, 호환성 검사, 성능 비교
- **GitHub Actions 자동화**: 매일 07:00 KST에 동기화 및 R2 백업 실행
- **구성 관리자**: 템플릿, 백업, 암호화 저장소, 환경 변수 내보내기 지원

## 🏗️ 시스템 아키텍처

```text
GitHub Actions (daily-r2-sync)
        │
        ▼
Sync Pipeline CLI ──▶ Markdown Processor ──▶ OpenAI Embeddings ──▶ Qdrant Cloud
        │                                              │
        └──────────────▶ Cloudflare R2 백업 ◀──────────┘
```

- **Sync Pipeline**: `src/sync_pipeline.py`에서 Git → 전처리 → 임베딩 → Qdrant 업서트를 오케스트레이션
- **Provider Factory**: OpenAI 임베딩 서비스와 Qdrant Cloud 서비스를 동적으로 생성
- **Migration Tools**: 클러스터 간 벡터 이동, 백업/복원, 성능 비교 제공 (`uv run python -m src.migration_tools ...`)
- **Configuration Manager**: 템플릿/프로필/백업/암호화를 관리 (`uv run python -m src.config_manager ...`)

## ✅ 요구 사항

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- OpenAI API Key (`OPENAI_API_KEY`)
- Qdrant Cloud 클러스터 URL 및 API Key (`QDRANT_CLOUD_URL`, `QDRANT_API_KEY`)
- Cloudflare R2 자격 증명 (선택: 백업 사용 시)

## ⚡ 빠른 시작

1. **저장소 클론 및 의존성 설치**
   ```bash
   git clone https://github.com/your-org/knue-policy-vectorizer.git
   cd knue-policy-vectorizer
   curl -LsSf https://astral.sh/uv/install.sh | sh  # uv 미설치 시
   uv sync
   uv pip install -e .
   ```

2. **환경 변수 설정** (`.env` 또는 셸 변수)
   ```bash
   export OPENAI_API_KEY="sk-live-..."
   export OPENAI_MODEL="text-embedding-3-small"
   export QDRANT_CLOUD_URL="https://abc123-example.aws.cloud.qdrant.io"
   export QDRANT_API_KEY="qdrant-api-key"
   export COLLECTION_NAME="knue_policies"
   export VECTOR_SIZE=1536
   export LOG_LEVEL=INFO
   # (선택) Cloudflare R2
   export CLOUDFLARE_ACCOUNT_ID="..."
   export CLOUDFLARE_R2_ACCESS_KEY_ID="..."
   export CLOUDFLARE_R2_SECRET_ACCESS_KEY="..."
   export CLOUDFLARE_R2_BUCKET="knue-policy-archive"
   export CLOUDFLARE_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
   ```

3. **테스트 실행**
```bash
uv run pytest -m "not slow"  # 빠른 단위 테스트만 실행
uv run pytest  # 모든 테스트 실행 (느린 테스트 포함)
    ```

4. **최초 동기화**
   ```bash
   uv run python -m src.sync_pipeline sync
   ```

## 🛠️ CLI 명령 요약

```bash
uv run python -m src.sync_pipeline list-providers
uv run python -m src.sync_pipeline configure
uv run python -m src.sync_pipeline show-config
uv run python -m src.sync_pipeline sync
uv run python -m src.sync_pipeline reindex
uv run python -m src.sync_pipeline health
uv run python -m src.sync_pipeline test-providers
uv run python -m src.sync_pipeline migrate   --from-embedding openai --from-vector qdrant_cloud   --to-embedding openai   --to-vector qdrant_cloud
```

- `configure`: 대화형으로 OpenAI/Qdrant Cloud 정보를 저장합니다 (환경 변수 업데이트는 수동 적용 필요).
- `test-providers`: OpenAI 및 Qdrant Cloud 연결을 각각 헬스 체크합니다.
- `migrate`: Qdrant Cloud 클러스터 간 스냅샷 전송, 백업/복원 및 성능 비교를 수행합니다.

## 📦 구성 템플릿 & 백업

- 기본 템플릿: `config/templates/openai-cloud.json`
- 템플릿 생성/목록/내보내기:
  ```bash
  uv run python -m src.config_manager list-templates
  uv run python -m src.config_manager export-template --name openai-cloud --format json
  ```
- 백업: `uv run python -m src.config_manager backup --name production`
- 복원: `uv run python -m src.config_manager restore --backup <path>`

## ☁️ GitHub Actions

- 워크플로우: `.github/workflows/daily-r2-sync.yml`
- 실행 주기: 매일 22:00 UTC (07:00 KST)
- 필수 시크릿: `OPENAI_API_KEY`, `OPENAI_MODEL`, `QDRANT_CLOUD_URL`, `QDRANT_API_KEY`, `CLOUDFLARE_*`
- 수동 실행: Actions 탭에서 **Run workflow** 버튼 사용
- 로컬 시뮬레이션: `uv run python -m src.sync_pipeline sync --log-level DEBUG`

## 🧰 Cloudflare R2 백업

- 백업 활성화: `CLOUDFLARE_R2_*` 환경 변수 설정 후 `uv run python -m src.sync_pipeline sync-cloudflare-r2`
- 소프트 삭제: `CLOUDFLARE_R2_SOFT_DELETE_ENABLED=true` 설정 시 `deleted/` 프리픽스에 보관
- 검증 스크립트: `uv run python scripts/verify_qdrant.py`

## 🔍 트러블슈팅

| 증상 | 확인 사항 | 해결책 |
| --- | --- | --- |
| OpenAI 401 Unauthorized | API Key/모델 이름 불일치 | `OPENAI_API_KEY`, `OPENAI_MODEL` 재확인 |
| Qdrant 403 Forbidden | API Key 혹은 URL 오타, TLS 문제 | `QDRANT_CLOUD_URL`, `QDRANT_API_KEY` 재발급 |
| Sync 실패 (네트워크) | GitHub Actions 제한 혹은 프록시 | 재시도 또는 Self-hosted Runner 고려 |
| Cloudflare 전송 오류 | R2 권한 누락 | `CLOUDFLARE_R2_ACCESS_KEY_ID/SECRET` IAM 정책 재확인 |

로그 레벨을 높이고 싶은 경우 `LOG_LEVEL=DEBUG` 설정 후 `uv run python -m src.sync_pipeline sync`로 재시도하세요.

## 🧪 테스트 전략 (TDD)

- 모든 변경 사항은 pytest 기반 단위/통합 테스트를 동반합니다.
- `tests/test_cli_providers.py`, `tests/test_config_env.py`, `tests/test_config_multi_provider.py`, `tests/test_migration_tools.py` 등에서 OpenAI + Qdrant Cloud 흐름을 검증합니다.
- 느린 통합 테스트는 `-m slow` 마커로 분리되어 있으며 CI에서는 기본적으로 제외됩니다.

## 🗺️ 프로젝트 구조

```
knue-policy-vectorizer/
├── src/                  # 파이프라인, 프로바이더, 구성 모듈
├── tests/                # pytest 스위트 (unit/integration)
├── config/templates/     # 환경 템플릿 (openai-cloud.json 등)
├── scripts/              # 운영 유틸리티 및 검증 스크립트
├── .spec/                # 수락 기준 및 로드맵
├── .agents/              # 정책/워크플로우/템플릿
└── README.md             # 현재 문서
```

## 📚 참고 자료

- [OpenAI Embeddings 가이드](https://platform.openai.com/docs/guides/embeddings)
- [Qdrant Cloud 문서](https://qdrant.tech/documentation/)
- [Cloudflare R2 소개](https://developers.cloudflare.com/r2/)
- [uv 공식 문서](https://docs.astral.sh/uv/)

---

> 시스템 변경 사항은 `.spec/sync-pipeline/project-roadmap.spec.md`와 `.agents/` 정책 문서를 함께 갱신하세요. 모든 배포는 GitHub Actions workflow 로그와 Cloudflare R2 백업 상태를 확인한 뒤 완료로 간주합니다.
