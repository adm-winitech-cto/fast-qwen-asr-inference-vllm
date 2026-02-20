# Docker Config Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Docker에서 재빌드 없이 외부 설정 파일/환경 변수만으로 서버 설정을 반영하고, 컨테이너 재시작 시 즉시 적용되게 한다.

**Architecture:** `app_config.py`를 추가해 기본 설정 파일 + 외부 오버라이드 파일 + 환경 변수 오버라이드를 병합 로드한다. `server.py`는 모든 환경 의존값을 `AppConfig`에서 읽도록 리팩토링하고, Docker/Makefile/README는 config 파일 마운트 기반 실행 흐름으로 정리한다.

**Tech Stack:** Python 3, FastAPI, dataclasses, JSON config, Docker, Makefile

---

### Task 1: Config 로더 TDD

**Files:**
- Create: `tests/test_app_config.py`
- Create: `app_config.py`
- Create: `config/default.json`

**Step 1: Write the failing test**
- 기본 설정 파일 로드
- 외부 override 파일 병합
- `APP_CFG__...` 환경 변수 오버라이드
- legacy env(`ASR_MODEL_NAME` 등) 오버라이드

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests.test_app_config -v`
Expected: FAIL (`ModuleNotFoundError` or missing functions)

**Step 3: Write minimal implementation**
- `load_app_config()` 구현
- dict deep merge 구현
- dataclass 변환 및 값 검증 구현

**Step 4: Run test to verify it passes**
Run: `python -m unittest tests.test_app_config -v`
Expected: PASS

### Task 2: 서버 설정 하드코딩 제거

**Files:**
- Modify: `server.py`

**Step 1: Write/adjust failing test if needed**
- Config 로더 테스트에서 필요한 키(스트리밍/동시성/모델) 보장 실패 케이스 추가

**Step 2: Run test to verify it fails**
Run: `python -m unittest tests.test_app_config -v`
Expected: FAIL for missing mapping/fields

**Step 3: Write minimal implementation**
- env 직접 참조 제거 및 `AppConfig` 참조로 대체
- uvicorn host/port/config 적용
- CORS/config 기반 처리
- warmup/init 파라미터 config화

**Step 4: Run tests**
Run: `python -m unittest tests.test_app_config -v`
Expected: PASS

### Task 3: Docker 연동 정리

**Files:**
- Modify: `Makefile`
- Modify: `README.md`
- Create: `config/docker.env`
- Modify: `Dockerfile` (필요 최소 변경)

**Step 1: Write failing expectation**
- 문서/실행 명령에서 외부 config mount 및 env-file 경로가 반영되지 않은 상태 확인

**Step 2: Implement**
- `docker run`에 config file mount + `APP_CONFIG_PATH` + `--env-file` 적용
- README에 config 변경 후 `docker restart`로 반영 절차 명시

**Step 3: Verify**
Run: `python -m unittest tests.test_app_config -v`
Run: `python -m py_compile server.py app_config.py benchmark.py`
Expected: all PASS, no syntax errors

### Task 4: 최종 검증

**Files:**
- Modify: `README.md` (검증 체크리스트 명시)

**Step 1: Verify requirement checklist**
- 컨테이너 기동
- config 변경 후 재시작 반영
- 설정 누락 시 안전 기본값/명확한 오류 처리

**Step 2: Final commands**
Run: `python -m unittest tests.test_app_config -v`
Run: `python -m py_compile server.py app_config.py`

