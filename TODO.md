# KNUE Policy Vectorizer - TODO List

## 프로젝트 개요

PRD.txt를 기반으로 한 **TDD(테스트 주도 개발)** 방식의 KNUE Policy Hub → Qdrant 동기화 파이프라인 구현

## 개발 원칙

- **TDD 접근**: 각 기능마다 테스트 먼저 작성 → 구현 → 테스트 통과 확인
- **단계별 검증**: 각 단계마다 실행 가능한 결과물과 테스트 결과 확인
- **점진적 구현**: 작은 단위로 구현하여 매 단계마다 동작 확인

---

## Phase 1: 프로젝트 기반 설정 ✅ COMPLETED

### ✅ 1.1 프로젝트 기본 구조 및 개발 환경 설정

- [x] 프로젝트 디렉토리 구조 생성 (src, tests, config, scripts)
- [x] requirements.txt 작성 (pytest, langchain, qdrant-client, GitPython 등)
- [x] pyproject.toml 설정 (pytest, black, isort, mypy 구성)
- [x] .gitignore 설정
- [x] 기본 로깅 설정 (structlog with colors)
- [x] 구성 관리 모듈 (config.py) 작성

**✅ 검증 완료**: 모든 의존성 설치 및 테스트 통과

### ✅ 1.2 Qdrant Docker Compose 파일 작성 및 테스트

- [x] docker-compose.qdrant.yml 작성 (Qdrant 서비스 구성)
- [x] Qdrant 구성 파일 작성 (config/qdrant.yml)
- [x] Qdrant 연결 테스트 스크립트 작성 (scripts/verify_qdrant.py)
- [x] 컬렉션 생성, 포인트 삽입/검색 테스트 완료
- [x] 헬스체크 및 1024차원 벡터 지원 확인

**✅ 검증 완료**: Qdrant 서비스 정상 작동, 모든 연결 테스트 통과

---

## Phase 2: Git 저장소 감시 기능 ✅ COMPLETED

### ✅ 2.1 Git 저장소 감시 기능 테스트 작성 (TDD)

- [x] `tests/test_git_watcher.py` 작성 (포괄적 테스트 스위트)
- [x] Git clone/pull 테스트 케이스 (mocking 포함)
- [x] HEAD 변경 감지 테스트 케이스
- [x] .md 파일 목록 추출 테스트 케이스
- [x] 파일별 커밋 정보 추출 테스트
- [x] 실제 Git 작업을 위한 통합 테스트

**✅ 검증 완료**: 8개 단위 테스트 모두 통과

### ✅ 2.2 Git 변경 감지 및 파일 목록 추출 구현

- [x] `src/git_watcher.py` 구현 (완전한 GitWatcher 클래스)
- [x] Git 저장소 클론/풀 기능
- [x] Markdown 파일 탐지 및 목록 생성
- [x] 커밋 간 변경 파일 감지 (추가/수정/삭제)
- [x] 파일별 커밋 정보 추출
- [x] GitHub URL 생성 기능
- [x] UTF-8 한국어 파일 지원

**✅ 검증 완료**: 모든 테스트 통과, TDD 사이클 완료

### ✅ 2.3 Git 감시 기능 통합 테스트 및 실행 확인

- [x] 실제 KNUE-Policy-Hub 저장소 테스트 완료
- [x] 100개 마크다운 정책 파일 탐지 확인
- [x] 한국어 파일명 및 내용 처리 확인
- [x] 통합 테스트 스크립트 작성 (scripts/test_git_watcher.py)
- [x] 실제 커밋 정보 및 파일 내용 읽기 검증

**✅ 검증 완료**: 실제 저장소에서 완벽한 동작 확인

---

## Phase 3: Markdown 전처리 기능 ✅ COMPLETED

### ✅ 3.1 Markdown 파일 전처리 테스트 작성 (TDD)

- [x] `tests/test_markdown_processor.py` 작성 (17개 포괄적 테스트)
- [x] YAML/TOML Frontmatter 제거 테스트
- [x] H1 제목 추출 및 폴백 테스트 (파일명 사용)
- [x] 콘텐츠 정리 및 구조 보존 테스트
- [x] 메타데이터 생성 테스트 (PRD 스키마 준수)
- [x] 문서 ID 계산 및 토큰 수 추정 테스트
- [x] 실제 KNUE 문서 형식 통합 테스트

**✅ 검증 완료**: 17개 테스트 모두 통과

### ✅ 3.2 Markdown 전처리 기능 구현

- [x] `src/markdown_processor.py` 구현 (완전한 MarkdownProcessor 클래스)
- [x] YAML/TOML frontmatter 제거 기능 (python-frontmatter + 수동 파싱)
- [x] 스마트 제목 추출 (H1 → 파일명 → 기본값 순서)
- [x] 콘텐츠 정리 (과도한 공백 제거, 마크다운 구조 보존)
- [x] PRD 스키마 준수 메타데이터 생성
- [x] 문서 ID 생성 (MD5 해시 기반)
- [x] 토큰 수 추정 및 길이 검증 (bge-m3 8192 토큰 제한)
- [x] 완전한 파이프라인 및 벡터화용 문서 생성

**✅ 검증 완료**: 모든 테스트 통과, 한국어 정책 문서 처리 최적화

---

## Phase 4: 임베딩 서비스 연동 ✅ COMPLETED

### ✅ 4.1 Ollama 임베딩 서비스 테스트 작성 (TDD)

- [x] `tests/test_embedding_service.py` 작성 (20개 포괄적 테스트)
- [x] Ollama 연결 테스트 (health check 포함)
- [x] bge-m3 모델 임베딩 테스트 (1024차원)
- [x] 토큰 길이 제한 테스트 (8192 토큰 제한)
- [x] 배치 처리 및 에러 핸들링 테스트
- [x] 한국어 텍스트 지원 테스트

**✅ 검증 완료**: 20개 단위 테스트 모두 통과

### ✅ 4.2 Ollama를 통한 임베딩 생성 기능 구현

- [x] `src/embedding_service.py` 구현 (완전한 EmbeddingService 클래스)
- [x] generate_embedding() 메서드 (단일 텍스트)
- [x] generate_embeddings_batch() 메서드 (배치 처리)
- [x] LangChain Ollama 연동 (langchain-ollama 사용)
- [x] 토큰 제한 검증 (tiktoken 활용)
- [x] 헬스체크 및 모델 정보 조회 기능
- [x] 포괄적 에러 핸들링 (EmbeddingError)

**✅ 검증 완료**: 모든 테스트 통과, TDD 사이클 완료

### ✅ 4.3 임베딩 생성 통합 테스트 및 실행 확인

- [x] 실제 markdown 문서로 임베딩 생성 테스트 완료
- [x] 임베딩 벡터 크기 (1024) 확인
- [x] 성능 측정 (평균 0.058초/임베딩)
- [x] 배치 처리 효율성 검증 (2.12x 성능 향상)
- [x] 한국어 정책 문서 처리 확인
- [x] MarkdownProcessor와 완전 통합 테스트

**✅ 검증 완료**: 통합 테스트 스크립트 실행, 모든 기능 정상 작동

---

## Phase 5: Qdrant 벡터 스토어 연동 ✅ COMPLETED

### ✅ 5.1 Qdrant 연동 테스트 작성 (TDD)

- [x] `tests/test_qdrant_service.py` 작성 (25개 포괄적 테스트)
- [x] 컬렉션 관리 테스트 (생성, 삭제, 존재 확인)
- [x] 포인트 CRUD 테스트 (삽입, 업데이트, 삭제, 조회)
- [x] 배치 처리 테스트 (다중 포인트 작업)
- [x] 메타데이터 스키마 검증 테스트
- [x] 검색 기능 테스트 (유사도 검색, 임계값)
- [x] 에러 핸들링 및 예외 상황 테스트

**✅ 검증 완료**: 25개 단위 테스트 모두 통과

### ✅ 5.2 Qdrant 벡터 스토어 연동 구현

- [x] `src/qdrant_service.py` 구현 (완전한 QdrantService 클래스)
- [x] 컬렉션 관리 메서드 (create, delete, exists, info)
- [x] 포인트 작업 메서드 (upsert, delete, search, get)
- [x] 배치 처리 지원 (upsert_points_batch, delete_points_batch)
- [x] 벡터 및 메타데이터 검증 로직
- [x] 포괄적 에러 핸들링 (QdrantError)
- [x] 헬스체크 및 연결 관리
- [x] 구조화된 로깅 및 모니터링

**✅ 검증 완료**: 모든 테스트 통과, TDD 사이클 완료

### ✅ 5.3 Qdrant 연동 통합 테스트 및 실행 확인

- [x] 완전한 파이프라인 통합 테스트 스크립트 작성
- [x] 한국어 정책 문서 3개 처리 확인
- [x] Markdown → 임베딩 → Qdrant 저장 파이프라인 검증
- [x] 실시간 검색 기능 테스트 (1.0000 정확도 달성)
- [x] 성능 측정 (임베딩: 0.129s/문서, 저장: 0.012s/문서)
- [x] 컬렉션 정보 및 데이터 일치성 확인
- [x] 자동화된 테스트 및 정리 프로세스

**✅ 검증 완료**: 통합 테스트 스크립트 실행, 모든 기능 정상 작동

---

## Phase 6: 전체 동기화 파이프라인 ✅ COMPLETED

### ✅ 6.1 전체 동기화 파이프라인 테스트 작성 (TDD)

- [x] `tests/test_sync_pipeline.py` 작성 (21개 포괄적 테스트)
- [x] End-to-end 동기화 테스트 (no changes, added, modified, deleted)
- [x] 추가/수정/삭제 시나리오 테스트 (혼합 변경사항 처리)
- [x] 에러 핸들링 테스트 (Git, 임베딩, Qdrant 오류)
- [x] 컬렉션 관리 테스트 (생성, 존재 확인)
- [x] 헬스체크 및 구성요소 초기화 테스트
- [x] 전체 재인덱싱 테스트

**✅ 검증 완료**: 21개 테스트 모두 통과

### ✅ 6.2 전체 파이프라인 구현

- [x] `src/sync_pipeline.py` 구현 (완전한 SyncPipeline 클래스)
- [x] SyncPipeline 클래스 구현 (lazy component initialization)
- [x] sync() 메서드 (커밋 추적 기반 증분 동기화)
- [x] reindex_all() 메서드 (전체 재인덱싱)
- [x] CLI 인터페이스 구현 (sync, reindex, health 명령어)
- [x] 포괄적 에러 핸들링 (SyncError 클래스)
- [x] 구조화된 로깅 및 진행 상황 보고
- [x] 토큰 길이 제한 검증 및 콘텐츠 필터링

**✅ 검증 완료**: 모든 테스트 통과, TDD 사이클 완료

### ✅ 6.3 전체 동기화 파이프라인 통합 테스트

- [x] 실제 KNUE Policy Hub 저장소 동기화 실행
- [x] 통합 테스트 스크립트 작성 (`scripts/test_full_sync_pipeline.py`)
- [x] 결과 로그 및 Qdrant 데이터 확인 (컬렉션 생성, 검색 기능)
- [x] 성능 측정 (헬스체크, 동기화, 검색 시간)
- [x] 자동 정리 프로세스 (테스트 데이터 삭제)
- [x] CLI 명령어 실제 동작 확인

**✅ 검증 완료**: 통합 테스트 스크립트 실행, 모든 기능 정상 작동

---

## Phase 7: Docker 및 배포 환경 ✅ COMPLETED

### ✅ 7.1 전체 시스템 Docker Compose 작성

- [x] 통합 docker-compose.yml 작성 (qdrant, indexer, ollama 로컬 실행)
- [x] Dockerfile 작성 (Python 애플리케이션, uv 패키지 매니저)
- [x] cron 설정 추가 (docker-compose.cron.yml, scripts/crontab)
- [x] 환경 변수 설정 (.env.example, .env.docker)
- [x] Docker 사용 가이드 작성 (DOCKER.md)

**✅ 검증 완료**: `docker-compose up` 실행 및 모든 서비스 정상 작동

### ✅ 7.2 Docker 환경에서 전체 시스템 테스트

- [x] Docker 환경에서 동기화 파이프라인 실행 (health check 통과)
- [x] 네트워크 연결 테스트 (Qdrant ↔ Indexer, Host ↔ Ollama)
- [x] 환경별 구성 테스트 (로컬, Docker 환경 변수)
- [x] 통합 테스트 스크립트 작성 (scripts/test_docker_environment.py)
- [x] Docker 이미지 빌드 및 컨테이너 실행 검증

**✅ 검증 완료**: Docker 환경에서 완전한 동기화 사이클 실행 확인

---

## Phase 8: 문서화 ✅ COMPLETED

### ✅ 8.1 README.md 작성

- [x] 프로젝트 개요 및 아키텍처 (시스템 다이어그램 포함)
- [x] 설치 및 설정 가이드 (단계별 상세 가이드)
- [x] 사용법 및 CLI 명령어 (예시 출력 포함)
- [x] Docker 실행 방법 (완전한 Docker 배포 가이드)
- [x] 트러블슈팅 가이드 (포괄적 문제 해결 방법)
- [x] 성능 벤치마크 및 최적화 팁
- [x] 개발 가이드 및 기여 방법

**✅ 검증 완료**: 포괄적인 README.md 작성 완료, DOCKER.md와 연계된 완전한 문서화

---

## 수용 기준 체크리스트

- [x] main HEAD 변경 없을 때 0 upsert/0 delete 동작 ✅
- [x] .md 파일 추가/수정 시 정확한 포인트 수 업데이트 ✅
- [x] .md 파일 삭제 시 Qdrant에서 해당 포인트 삭제 ✅
- [x] 벡터 차원 1024 유지 ✅
- [x] 실패 파일에 대한 상세 로그 및 재시도 로직 ✅
- [x] Docker Compose로 외부 의존성 없이 실행 가능 ✅

**모든 수용 기준을 만족했습니다!** 🎉

---

## 진행 상황 추적

### ✅ **완료된 Phase (8/8)**

- **Phase 1**: 프로젝트 기반 설정 ✅ COMPLETED
- **Phase 2**: Git 저장소 감시 기능 ✅ COMPLETED
- **Phase 3**: Markdown 전처리 기능 ✅ COMPLETED
- **Phase 4**: 임베딩 서비스 연동 ✅ COMPLETED
- **Phase 5**: Qdrant 벡터 스토어 연동 ✅ COMPLETED
- **Phase 6**: 전체 동기화 파이프라인 ✅ COMPLETED
- **Phase 7**: Docker 및 배포 환경 ✅ COMPLETED
- **Phase 8**: 문서화 ✅ COMPLETED

### 🎉 **프로젝트 완료**: 모든 Phase 성공적으로 완료

**최종 상태**: 프로덕션 배포 준비 완료

### 📊 **전체 진행률**: 21/21 tasks completed (100%)

### 🎯 **주요 성과**

- ✅ 완전한 프로젝트 구조 및 개발 환경 구축 (uv, pytest, VS Code 통합)
- ✅ Qdrant 벡터 데이터베이스 Docker 환경 구성 (1024차원 bge-m3 지원)
- ✅ 실제 KNUE Policy Hub 저장소 (100개 마크다운 파일) 처리 가능한 Git 감시자 구현
- ✅ 포괄적 Markdown 전처리 파이프라인 (frontmatter 제거, 제목 추출, 메타데이터 생성)
- ✅ 완전한 Ollama 임베딩 서비스 연동 (bge-m3 모델, 1024차원, 배치 처리 지원)
- ✅ 완전한 Qdrant 벡터 스토어 연동 (컬렉션 관리, CRUD 작업, 검색 기능)
- ✅ **완전한 동기화 파이프라인 구현** (증분 동기화, 전체 재인덱싱, CLI 인터페이스)
- ✅ TDD 방법론 적용으로 모든 기능 테스트 커버리지 확보 (104개 테스트 통과)
- ✅ 한국어 정책 문서 UTF-8 처리 및 토큰 길이 검증 완료
- ✅ 고성능 임베딩 생성 (평균 0.129초/임베딩, 저장 0.012초/문서)
- ✅ End-to-end 파이프라인 통합 테스트 (Git → Markdown → 임베딩 → Qdrant → 검색)
- ✅ **완전한 Docker 배포 환경 구축** (docker-compose 2종, 환경 변수, cron 스케줄링)
- ✅ **완전한 프로젝트 문서화** (README.md, DOCKER.md, 사용자 가이드, 트러블슈팅)
- ✅ **프로덕션 준비 완료**: CLI 명령어, 에러 핸들링, 구조화된 로깅, Docker 컨테이너화

## 🎉 최종 결과

**KNUE Policy Vectorizer 프로젝트가 성공적으로 완료되었습니다!**

### 📊 **최종 통계**

- **총 104개의 테스트** 모두 통과 ✅
- **8개 Phase 21개 작업** 모두 완료 ✅
- **TDD 방법론** 완전 적용 ✅
- **프로덕션 배포** 준비 완료 ✅

### 🐳 **Docker 환경**

- **docker-compose.yml**: 개발/기본 프로덕션용 (Qdrant + 자동 동기화)
- **docker-compose.cron.yml**: 고급 프로덕션용 (cron 스케줄링)
- **Dockerfile**: 최적화된 Python 컨테이너 (uv 패키지 매니저)

### 📚 **문서화**

- **README.md**: 포괄적 사용자 가이드 (설치, 사용법, 트러블슈팅)
- **DOCKER.md**: 상세 Docker 배포 가이드
- **TODO.md**: 완전한 개발 진행 기록

### 🏆 **핵심 성과**

- **한국어 정책 문서 자동 벡터화**: Git → 전처리 → 임베딩 → 저장
- **고성능 처리**: 평균 0.129초/임베딩, 0.012초/저장
- **완전 자동화**: 증분 동기화, 스케줄링, 에러 핸들링
- **프로덕션 품질**: 104개 테스트, 구조화된 로깅, 모니터링

이제 **KNUE Policy Hub의 100개 마크다운 정책 문서**를 실시간으로 벡터화하여 **의미 검색 및 RAG 시스템**에 활용할 수 있습니다!

각 단계마다 테스트 실행 결과와 실제 동작 확인을 통해 진행 상황을 검증했습니다.

---

## Phase 9: Multi-Provider Support Enhancement 🚀 NEW

### 9.1 Multi-Provider Configuration Design

- [ ] Create provider enums for embedding services (OLLAMA, OPENAI) and vector databases (QDRANT_LOCAL, QDRANT_CLOUD)
- [ ] Design abstract interfaces for EmbeddingServiceInterface and VectorServiceInterface  
- [ ] Extend Config class to support provider selection via environment variables
- [ ] Add provider-specific configuration validation and parameter handling
- [ ] Design provider factory pattern for dynamic service instantiation

### 9.2 OpenAI Embedding Service Implementation

- [ ] Create OpenAIEmbeddingService class implementing EmbeddingServiceInterface
- [ ] Add OpenAI API client integration with proper authentication
- [ ] Implement text-embedding-3-small and text-embedding-3-large model support
- [ ] Add token counting and validation for OpenAI models (8192/8191 token limits)
- [ ] Handle OpenAI-specific rate limiting and error responses
- [ ] Add batch processing support for OpenAI embeddings

### 9.3 Qdrant Cloud Support Implementation

- [ ] Extend QdrantService to support cloud authentication (API keys)
- [ ] Add Qdrant Cloud URL configuration and connection handling
- [ ] Implement cloud-specific collection management and security settings
- [ ] Add support for Qdrant Cloud clusters and regions
- [ ] Handle cloud-specific rate limits and connection pooling
- [ ] Add health check and connectivity validation for cloud instances

### 9.4 CLI Provider Selection Interface

- [ ] Add `configure` CLI command for interactive provider setup
- [ ] Implement `--provider-embedding` and `--provider-vector` CLI options
- [ ] Add `list-providers` command to show available configurations
- [ ] Create `test-providers` command for connectivity validation
- [ ] Add provider configuration status display in health check
- [ ] Implement configuration persistence and environment file generation

### 9.5 Environment Variable Configuration Extension

- [ ] Add EMBEDDING_PROVIDER and VECTOR_PROVIDER environment variables
- [ ] Extend Config.from_env() to handle provider-specific configurations
- [ ] Add OpenAI-specific variables (OPENAI_API_KEY, OPENAI_MODEL, OPENAI_BASE_URL)
- [ ] Add Qdrant Cloud variables (QDRANT_API_KEY, QDRANT_CLUSTER_URL)
- [ ] Support legacy environment variable mapping for backward compatibility
- [ ] Add configuration validation and meaningful error messages

### 9.6 Migration and Compatibility Tools

- [ ] Create migration script for transferring vectors between providers
- [ ] Add compatibility check for vector dimensions across providers
- [ ] Implement backup and restore functionality for provider switching
- [ ] Add configuration migration tools for environment setup
- [ ] Create provider performance comparison utilities
- [ ] Add rollback capabilities for failed migrations

### 9.7 Multi-Provider Testing Suite

- [ ] Write comprehensive unit tests for provider factories and interfaces
- [ ] Add integration tests for OpenAI embedding service
- [ ] Create Qdrant Cloud integration tests (with test credentials)
- [ ] Implement provider switching tests and compatibility validation
- [ ] Add performance benchmarking tests across providers
- [ ] Create end-to-end tests for complete multi-provider workflows

### 9.8 Configuration Management Enhancement

- [ ] Add configuration validation and schema checking
- [ ] Implement configuration templates for common provider setups
- [ ] Add configuration backup and versioning
- [ ] Create configuration import/export functionality
- [ ] Add environment-specific configuration profiles
- [ ] Implement configuration security and credential management

### 9.9 Documentation and User Guide Updates

- [ ] Update README.md with multi-provider setup instructions
- [ ] Create provider comparison guide (features, performance, costs)
- [ ] Add step-by-step provider migration guide
- [ ] Update Docker documentation for multi-provider deployments
- [ ] Create troubleshooting guide for provider-specific issues
- [ ] Add configuration examples for different deployment scenarios

## Multi-Provider Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Sync Pipeline                            │
└─────────────────┬───────────────────────┬───────────────────┘
                  │                       │
         ┌────────▼────────┐     ┌────────▼────────┐
         │ Embedding       │     │ Vector          │
         │ Service Factory │     │ Service Factory │
         └─────────────────┘     └─────────────────┘
                  │                       │
       ┌──────────┼──────────┐           ┌┼─────────────┐
       │          │          │           ││             │
   ┌───▼───┐ ┌───▼───┐ ┌────▼────┐ ┌───▼▼───┐ ┌──────▼──────┐
   │Ollama │ │OpenAI │ │  Future │ │ Qdrant │ │ Qdrant      │
   │ bge-m3│ │ text- │ │Provider │ │ Local  │ │ Cloud       │
   │       │ │embed-3│ │         │ │        │ │             │
   └───────┘ └───────┘ └─────────┘ └────────┘ └─────────────┘
```

## Key Environment Variables for Multi-Provider Setup

```bash
# Provider Selection
EMBEDDING_PROVIDER=ollama|openai
VECTOR_PROVIDER=qdrant_local|qdrant_cloud

# Ollama Configuration (existing)
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=bge-m3

# OpenAI Configuration (new)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=text-embedding-3-small|text-embedding-3-large
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional for custom endpoints

# Qdrant Local Configuration (existing)
QDRANT_URL=http://localhost:6333

# Qdrant Cloud Configuration (new)
QDRANT_CLOUD_URL=https://your-cluster.qdrant.tech
QDRANT_API_KEY=your-api-key
QDRANT_CLUSTER_REGION=us-east-1  # Optional
```

## CLI Usage Examples

```bash
# Configure providers interactively
uv run python -m src.sync_pipeline configure

# Set providers via CLI
uv run python -m src.sync_pipeline --embedding-provider openai --vector-provider qdrant_cloud sync

# List available providers
uv run python -m src.sync_pipeline list-providers

# Test provider connectivity
uv run python -m src.sync_pipeline test-providers

# Migrate between providers
uv run python -m src.sync_pipeline migrate --from ollama,qdrant_local --to openai,qdrant_cloud
```

## Expected Benefits

1. **Flexibility**: Choose optimal providers for different use cases and environments
2. **Scalability**: Use cloud providers for production deployments
3. **Cost Optimization**: Select cost-effective providers based on usage patterns
4. **Redundancy**: Support multiple providers for high availability
5. **Future-Proofing**: Easy integration of new embedding models and vector databases

## Implementation Priority

**High Priority** (Core functionality):
- Provider enums and interfaces (9.1)
- OpenAI embedding service (9.2)
- Basic CLI provider selection (9.4)
- Environment variable extension (9.5)

**Medium Priority** (Enhanced features):
- Qdrant Cloud support (9.3)
- Migration tools (9.6)
- Comprehensive testing (9.7)

**Lower Priority** (Polish and documentation):
- Advanced configuration management (9.8)
- Documentation updates (9.9)
