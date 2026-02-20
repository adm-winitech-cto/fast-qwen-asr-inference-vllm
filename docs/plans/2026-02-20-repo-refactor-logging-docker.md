# Repository Refactor, Logging, and Docker Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the repository into clear source/config/docker/log directories, unify application logging with rotation, and enforce Docker log size limits without breaking existing server behavior.

**Architecture:** Move runtime modules into a package under `src/fast_qwen_asr`, keep thin root-level compatibility wrappers, and centralize startup/config/logging concerns in dedicated modules. Add docker-compose configuration with explicit container logging policy and ensure app logs are persisted to project `logs/`.

**Tech Stack:** Python 3, FastAPI, grpcio, standard `logging` + `RotatingFileHandler`, Docker/Docker Compose, unittest/pytest-compatible tests.

---

### Task 1: Define target package layout and runtime entrypoints

**Files:**
- Create: `src/fast_qwen_asr/__init__.py`
- Create: `src/fast_qwen_asr/api/http_server.py`
- Create: `src/fast_qwen_asr/api/grpc_runtime.py`
- Create: `src/fast_qwen_asr/api/grpc_service.py`
- Create: `src/fast_qwen_asr/config/app_config.py`
- Create: `src/fast_qwen_asr/core/runtime_state.py`
- Create: `src/fast_qwen_asr/core/model_loader.py`
- Create: `src/fast_qwen_asr/services/audio_decoder.py`
- Create: `src/fast_qwen_asr/utils/inference.py`
- Create: `src/fast_qwen_asr/utils/server.py`
- Create: `src/fast_qwen_asr/logging/setup.py`
- Modify: `server.py`
- Modify: `grpc_runtime.py`
- Modify: `grpc_service.py`
- Modify: `app_config.py`
- Modify: `server_runtime.py`
- Modify: `inference_utils.py`
- Modify: `server_utils.py`

**Step 1: Write failing import-path tests**
- Add test coverage that imports new package modules and verifies root wrappers still export expected APIs.

**Step 2: Run tests to verify RED**
Run: `pytest -q tests/test_module_layout.py`
Expected: FAIL for missing package modules.

**Step 3: Write minimal implementation**
- Add package files and move/refactor existing logic with clear responsibility boundaries.
- Keep root wrappers for backward compatibility.

**Step 4: Run tests to verify GREEN**
Run: `pytest -q tests/test_module_layout.py`
Expected: PASS.

### Task 2: Centralize and harden logging

**Files:**
- Create: `src/fast_qwen_asr/logging/setup.py`
- Modify: `src/fast_qwen_asr/config/app_config.py`
- Modify: `config/default.json`
- Modify: `config/runtime.json`
- Test: `tests/test_logging_setup.py`

**Step 1: Write failing tests for logger config and `logs/` creation**
- Validate timestamp/level/logger/message format and rotating file handler parameters.

**Step 2: Run tests to verify RED**
Run: `pytest -q tests/test_logging_setup.py`
Expected: FAIL until logging module is implemented.

**Step 3: Write minimal implementation**
- Add logging config dataclass fields.
- Create function to initialize root logger, ensure `logs/` exists, and attach console + rotating file handlers.

**Step 4: Run tests to verify GREEN**
Run: `pytest -q tests/test_logging_setup.py`
Expected: PASS.

### Task 3: Remove duplication and wire app runtime to shared modules

**Files:**
- Modify: `src/fast_qwen_asr/api/http_server.py`
- Modify: `src/fast_qwen_asr/api/grpc_service.py`
- Modify: `src/fast_qwen_asr/services/audio_decoder.py`
- Modify: `benchmark.py`
- Modify: `grpc_batch_benchmark.py`
- Modify: `client-grpc.py`

**Step 1: Write failing behavior tests for shared utilities (if absent)**
- Add/extend tests around audio decoding error handling and model-name checks.

**Step 2: Run tests to verify RED**
Run: `pytest -q tests/test_server_utils.py tests/test_inference_utils.py`
Expected: At least one failure due to moved imports until wrappers are updated.

**Step 3: Write minimal implementation**
- Move shared logic to utilities/services modules.
- Replace duplicated constants/magic numbers with config-driven constants.
- Remove dead imports and redundant branches.

**Step 4: Run tests to verify GREEN**
Run: `pytest -q tests/test_server_utils.py tests/test_inference_utils.py tests/test_grpc_service.py`
Expected: PASS.

### Task 4: Docker and compose logging policy

**Files:**
- Create: `docker/docker-compose.yml`
- Modify: `Dockerfile`
- Modify: `Makefile`
- Modify: `README.md`

**Step 1: Write failing config test/validation check**
- Add test or script validation asserting compose contains `json-file`, `max-size`, `max-file`.

**Step 2: Run check to verify RED**
Run: `rg -n "max-size|max-file|logging:" docker/docker-compose.yml`
Expected: FAIL (file missing before implementation).

**Step 3: Write minimal implementation**
- Add compose service and Docker logging limits.
- Route app output level default to INFO.
- Mount project `logs/` path.

**Step 4: Run check to verify GREEN**
Run: `rg -n "max-size|max-file|logging:" docker/docker-compose.yml`
Expected: Matches present.

### Task 5: Update tests and docs to new structure

**Files:**
- Modify: `tests/test_*.py` (imports where needed)
- Modify: `README.md`
- Create: `logs/.gitkeep`

**Step 1: Write/adjust failing tests for config and wrappers**
- Ensure all tests import from stable API paths.

**Step 2: Run tests to verify RED**
Run: `pytest -q`
Expected: Failures until all imports are aligned.

**Step 3: Write minimal implementation**
- Adjust tests to package imports and keep compatibility wrappers.
- Document new directory structure and run commands.

**Step 4: Run tests to verify GREEN**
Run: `pytest -q`
Expected: PASS.

### Task 6: Verification and readiness

**Files:**
- N/A (verification only)

**Step 1: Run quality checks**
Run: `python -m compileall src tests`
Expected: no syntax errors.

**Step 2: Run full tests**
Run: `pytest -q`
Expected: all tests pass.

**Step 3: Validate docker files**
Run: `docker compose -f docker/docker-compose.yml config`
Expected: valid compose output.

**Step 4: Commit**
```bash
git add .
git commit -m "refactor: reorganize source layout and harden logging/docker ops"
```
