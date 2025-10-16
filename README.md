# KNUE Policy Vectorizer

한국교원대학교 정책 문서를 OpenAI 임베딩과 Qdrant Cloud에 동기화하는 자동화 파이프라인입니다.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](./tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Vector DB](https://img.shields.io/badge/vector%20db-Qdrant%20Cloud-orange)](https://qdrant.tech)
[![Embeddings](https://img.shields.io/badge/embeddings-OpenAI%20text--embedding--3-purple)](https://platform.openai.com/docs)
[![Workflow](https://img.shields.io/badge/automation-GitHub%20Actions-black)](.github/workflows/daily-r2-sync.yml)
[![TDD](https://img.shields.io/badge/development-TDD-green)](./tests)

## 📋 개요

[한국교원대학교 정책 Hub](https://github.com/kadragon/KNUE-Policy-Hub)의 마크다운 문서를 자동으로 수집·전처리 → OpenAI 임베딩 변환 → Qdrant Cloud 저장. Cloudflare R2 백업 및 GitHub Actions 자동화 지원.

### 🎯 주요 기능

- **증분 동기화**: Git 변경 감시로 변경된 문서만 처리
- **스마트 청킹**: 800/200 토큰 단위 문서 분할 및 메타데이터 생성
- **벡터 저장소**: OpenAI 임베딩 + Qdrant Cloud 통합
- **자동 백업**: Cloudflare R2 + 소프트 삭제 관리
- **일정 실행**: GitHub Actions로 매일 07:00 KST 동기화

## 🏗️ 시스템 아키텍처

```text
GitHub Actions (daily-r2-sync)
        │
        ▼
Sync Pipeline CLI ──▶ Markdown Processor ──▶ OpenAI Embeddings ──▶ Qdrant Cloud
        │                                              │
        └──────────────▶ Cloudflare R2 백업 ◀──────────┘
```

- **Sync Pipeline** (`src/sync_pipeline.py`): Git → 전처리 → 임베딩 → Qdrant 업서트 오케스트레이션
- **Provider Factory**: OpenAI + Qdrant Cloud 동적 초기화
- **Migration Tools**: 클러스터 간 벡터 이동 및 검증

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

2. **필수 환경 변수 설정** (`.env.example` 참고)
    ```bash
    export OPENAI_API_KEY="sk-live-..."
    export OPENAI_MODEL="text-embedding-3-small"
    export QDRANT_CLOUD_URL="https://abc123-example.aws.cloud.qdrant.io"
    export QDRANT_API_KEY="qdrant-api-key"
    ```

3. **테스트 실행**
    ```bash
    uv run pytest  # 전체 테스트
    uv run pytest -m "not slow"  # 빠른 테스트만
    ```

4. **초기 동기화**
    ```bash
    uv run python -m src.sync_pipeline sync
    ```

## 🛠️ CLI 주요 명령

| 명령 | 설명 |
|------|------|
| `sync` | 정책 문서 수집 → 전처리 → 임베딩 → 저장 |
| `reindex` | 전체 컬렉션 재색인 |
| `health` | 시스템 헬스 체크 |
| `test-providers` | OpenAI/Qdrant 연결 검증 |
| `migrate` | Qdrant 클러스터 간 마이그레이션 |

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

## 🧪 테스트 (TDD)

- pytest 기반 단위/통합 테스트
- 느린 테스트는 `-m slow` 마커로 분리 (CI 제외)

## 🗺️ 프로젝트 구조

```
knue-policy-vectorizer/
├── src/              # 파이프라인, 프로바이더, 구성
├── tests/            # pytest 스위트
├── scripts/          # 유틸리티 및 검증
├── .spec/            # 수락 기준
├── .agents/          # 정책/워크플로우
└── .github/          # 자동화 워크플로우
```

## 📚 참고 자료

- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Qdrant Cloud](https://qdrant.tech/documentation/)
- [Cloudflare R2](https://developers.cloudflare.com/r2/)
- [uv](https://docs.astral.sh/uv/)
