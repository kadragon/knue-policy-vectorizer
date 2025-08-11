# KNUE Policy Vectorizer - TODO List

## 프로젝트 개요
PRD.txt를 기반으로 한 **TDD(테스트 주도 개발)** 방식의 KNUE Policy Hub → Qdrant 동기화 파이프라인 구현

## 개발 원칙
- **TDD 접근**: 각 기능마다 테스트 먼저 작성 → 구현 → 테스트 통과 확인
- **단계별 검증**: 각 단계마다 실행 가능한 결과물과 테스트 결과 확인
- **점진적 구현**: 작은 단위로 구현하여 매 단계마다 동작 확인

---

## Phase 1: 프로젝트 기반 설정

### ✅ 1.1 프로젝트 기본 구조 및 개발 환경 설정
- [ ] 프로젝트 디렉토리 구조 생성
- [ ] requirements.txt 작성 (pytest, langchain, qdrant-client, GitPython 등)
- [ ] pytest 설정 파일 (pytest.ini 또는 pyproject.toml)
- [ ] .gitignore 설정
- [ ] 기본 로깅 설정

**검증 방법**: `pytest --version`, `pip install -r requirements.txt` 실행 확인

### ✅ 1.2 Qdrant Docker Compose 파일 작성 및 테스트
- [ ] docker-compose.yml 작성 (Qdrant 서비스만)
- [ ] Qdrant 연결 테스트 스크립트 작성
- [ ] `docker-compose up -d` 실행 및 연결 확인

**검증 방법**: `curl http://localhost:6333/health` 성공 응답 확인

---

## Phase 2: Git 저장소 감시 기능

### ✅ 2.1 Git 저장소 감시 기능 테스트 작성 (TDD)
- [ ] `tests/test_git_watcher.py` 작성
- [ ] Git clone/pull 테스트 케이스
- [ ] HEAD 변경 감지 테스트 케이스  
- [ ] .md 파일 목록 추출 테스트 케이스

**검증 방법**: `pytest tests/test_git_watcher.py -v` 실행 (실패 예상)

### ✅ 2.2 Git 변경 감지 및 파일 목록 추출 구현
- [ ] `src/git_watcher.py` 구현
- [ ] GitRepository 클래스 구현
- [ ] clone_or_pull() 메서드
- [ ] get_changed_files() 메서드
- [ ] get_md_files() 메서드

**검증 방법**: 테스트 통과 확인

### ✅ 2.3 Git 감시 기능 통합 테스트 및 실행 확인
- [ ] 실제 KNUE-Policy-Hub 저장소로 테스트
- [ ] 변경사항 감지 동작 확인
- [ ] 로깅 출력 확인

**검증 방법**: 실제 저장소 clone 및 파일 목록 출력 확인

---

## Phase 3: Markdown 전처리 기능

### ✅ 3.1 Markdown 파일 전처리 테스트 작성
- [ ] `tests/test_markdown_processor.py` 작성
- [ ] Frontmatter 제거 테스트
- [ ] H1 제목 추출 테스트
- [ ] 메타데이터 생성 테스트

**검증 방법**: `pytest tests/test_markdown_processor.py -v` 실행 (실패 예상)

### ✅ 3.2 Markdown 전처리 기능 구현
- [ ] `src/markdown_processor.py` 구현
- [ ] MarkdownProcessor 클래스 구현
- [ ] process_markdown() 메서드
- [ ] extract_title() 메서드
- [ ] generate_metadata() 메서드

**검증 방법**: 테스트 통과 및 샘플 .md 파일 처리 결과 확인

---

## Phase 4: 임베딩 서비스 연동

### ✅ 4.1 Ollama 임베딩 서비스 테스트 작성
- [ ] `tests/test_embedding_service.py` 작성
- [ ] Ollama 연결 테스트
- [ ] bge-m3 모델 임베딩 테스트
- [ ] 토큰 길이 제한 테스트

**검증 방법**: `pytest tests/test_embedding_service.py -v` 실행 (실패 예상)

### ✅ 4.2 Ollama를 통한 임베딩 생성 기능 구현
- [ ] `src/embedding_service.py` 구현
- [ ] EmbeddingService 클래스 구현
- [ ] generate_embedding() 메서드
- [ ] LangChain OllamaEmbeddings 연동

**검증 방법**: 테스트 통과 및 샘플 텍스트 임베딩 생성 확인

### ✅ 4.3 임베딩 생성 통합 테스트 및 실행 확인
- [ ] 실제 markdown 문서로 임베딩 생성 테스트
- [ ] 임베딩 벡터 크기 (1024) 확인
- [ ] 성능 측정 (처리 시간)

**검증 방법**: 실제 문서 임베딩 결과 및 차원 확인

---

## Phase 5: Qdrant 벡터 스토어 연동

### ✅ 5.1 Qdrant 연동 테스트 작성
- [ ] `tests/test_qdrant_service.py` 작성
- [ ] 컬렉션 생성 테스트
- [ ] 포인트 삽입/업데이트 테스트
- [ ] 포인트 삭제 테스트
- [ ] 메타데이터 스키마 테스트

**검증 방법**: `pytest tests/test_qdrant_service.py -v` 실행 (실패 예상)

### ✅ 5.2 Qdrant 벡터 스토어 연동 구현
- [ ] `src/qdrant_service.py` 구현
- [ ] QdrantService 클래스 구현
- [ ] create_collection() 메서드
- [ ] upsert_points() 메서드
- [ ] delete_points() 메서드

**검증 방법**: 테스트 통과 확인

### ✅ 5.3 Qdrant 연동 통합 테스트 및 실행 확인
- [ ] 실제 임베딩 데이터로 Qdrant 저장 테스트
- [ ] Qdrant 대시보드에서 데이터 확인
- [ ] 검색 쿼리 테스트

**검증 방법**: Qdrant 웹 UI에서 컬렉션 및 포인트 확인

---

## Phase 6: 전체 동기화 파이프라인

### ✅ 6.1 전체 동기화 파이프라인 테스트 작성
- [ ] `tests/test_sync_pipeline.py` 작성
- [ ] End-to-end 동기화 테스트
- [ ] 추가/수정/삭제 시나리오 테스트
- [ ] 에러 핸들링 테스트

**검증 방법**: `pytest tests/test_sync_pipeline.py -v` 실행 (실패 예상)

### ✅ 6.2 전체 파이프라인 구현
- [ ] `src/sync_pipeline.py` 구현
- [ ] SyncPipeline 클래스 구현
- [ ] sync() 메서드 (메인 동기화 로직)
- [ ] reindex_all() 메서드
- [ ] CLI 인터페이스 구현

**검증 방법**: 테스트 통과 확인

### ✅ 6.3 전체 동기화 파이프라인 통합 테스트
- [ ] 실제 저장소로 전체 동기화 실행
- [ ] 결과 로그 및 Qdrant 데이터 확인
- [ ] 성능 측정 및 최적화

**검증 방법**: 전체 동기화 완료 및 데이터 일치성 확인

---

## Phase 7: Docker 및 배포 환경

### ✅ 7.1 전체 시스템 Docker Compose 작성
- [ ] 통합 docker-compose.yml 작성 (ollama, qdrant, indexer)
- [ ] Dockerfile 작성 (Python 애플리케이션)
- [ ] cron 설정 추가
- [ ] 환경 변수 설정

**검증 방법**: `docker-compose up` 실행 확인

### ✅ 7.2 Docker 환경에서 전체 시스템 테스트
- [ ] Docker 환경에서 동기화 파이프라인 실행
- [ ] cron 작업 동작 확인
- [ ] 로그 수집 및 모니터링

**검증 방법**: Docker 환경에서 완전한 동기화 사이클 실행 확인

---

## Phase 8: 문서화

### ✅ 8.1 README.md 작성
- [ ] 프로젝트 개요 및 아키텍처
- [ ] 설치 및 설정 가이드
- [ ] 사용법 및 CLI 명령어
- [ ] Docker 실행 방법
- [ ] 트러블슈팅 가이드

**검증 방법**: README 내용대로 신규 환경에서 설치/실행 가능한지 확인

---

## 수용 기준 체크리스트

- [ ] main HEAD 변경 없을 때 0 upsert/0 delete 동작
- [ ] .md 파일 추가/수정 시 정확한 포인트 수 업데이트
- [ ] .md 파일 삭제 시 Qdrant에서 해당 포인트 삭제
- [ ] 벡터 차원 1024 유지
- [ ] 실패 파일에 대한 상세 로그 및 재시도 로직
- [ ] Docker Compose로 외부 의존성 없이 실행 가능

---

## 진행 상황 추적

**현재 진행중**: Phase 1 - 프로젝트 기반 설정

**다음 단계**: 프로젝트 기본 구조 및 개발 환경 설정 완료 후 Qdrant Docker Compose 작성

각 단계마다 테스트 실행 결과와 실제 동작 확인을 통해 진행 상황을 검증합니다.