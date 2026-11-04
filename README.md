# Fast Qwen3-ASR Inference Server

FastAPI + vLLM 기반 Qwen3-ASR 추론 서버입니다.
HTTP `/transcribe`, WebSocket `/transcribe-streaming`, gRPC(`STTService`)를 제공합니다.

## 프로젝트 구조

- `src/fast_qwen_asr/`: 애플리케이션 소스
- `config/`: 런타임 설정(`default.json`, `runtime.json`, `offline.runtime.json`)
- `docker/`: Dockerfile, compose 파일
- `tests/`: 테스트 코드
- `logs/`: 애플리케이션 로그(`app.log`, 로테이션)

## 빠른 시작 (온라인 환경)

1. 이미지 빌드/기동

```bash
make up
```

2. 헬스체크

```bash
make health
```

3. 로그 확인

```bash
make logs
```

기본 포트:
- HTTP: `8907`
- gRPC: `50051`

## 폐쇄망 배포 가이드

폐쇄망에서는 `오프라인 이미지 + 외부 모델 볼륨` 방식으로 운영합니다.
모델은 이미지에 내장하지 않으므로, 모델 교체 시 이미지 재빌드가 필요 없습니다.

### 1) 온라인 환경에서 준비

```bash
# 베이스 이미지 빌드 (DGX Spark / GB10 Blackwell, aarch64 · CUDA 12.8 툴킷, 13.0 드라이버)
# 이 태그는 공개 레지스트리가 아니라 로컬에서 직접 빌드합니다.
make build          # -> qwenllm/qwen3-asr:cu128-sbsa

# 오프라인 실행용 앱 이미지 빌드
make build-offline

# 배포용 tar 생성
make save-offline
```

생성 파일:
- `dist/qwen3-asr-offline.tar`

### 2) 폐쇄망으로 반입

아래를 폐쇄망 서버로 전달합니다.
- `dist/qwen3-asr-offline.tar`
- 모델 디렉토리 (예: `/data/asr-models/1.7b-v2-cp47728`)
- 필요 시 설정 파일 (`config/offline.runtime.json`)

### 3) 폐쇄망 서버에서 로드/실행

```bash
# 이미지 로드
make load-offline

# 실행 (기본 모델 경로/디렉토리명 사용)
make up-offline
```

기본값:
- `OFFLINE_MODELS_DIR=./models`
- `OFFLINE_MODELS_MOUNT=/opt/models`
- `OFFLINE_ASR_MODEL_DIR=1.7b-v2-cp47728`
- `OFFLINE_ALIGNER_MODEL_DIR=0.6b-v3-cp22426`

실행 시 내부적으로 아래가 적용됩니다.
- `-v ./models:/opt/models:ro`
- `APP_CFG__MODELS__ASR_MODEL_PATH=/opt/models/<ASR_MODEL_DIR>`
- `APP_CFG__MODELS__ALIGNER_MODEL_PATH=/opt/models/<ALIGNER_MODEL_DIR>`
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- Docker 로그 제한: `--log-opt max-size=10m --log-opt max-file=3`

### 4) 폐쇄망 기동 확인 (운영 절차)

1. 컨테이너 기동

```bash
make up-offline
```

참고:
- `make up-offline`는 마지막에 `docker logs -f qwen3-asr-offline`를 실행하므로 로그를 계속 팔로우합니다.
- 기동 로그 확인 후 `Ctrl+C`로 빠져나와도 컨테이너는 계속 실행됩니다.

2. 헬스체크 확인

```bash
curl -s http://localhost:8907/health | jq .
```

기대 결과:
- 초기에는 `"status": "loading_models"`가 나올 수 있음 (정상)
- 모델 로딩 완료 후 `"status": "ready"`여야 정상
- 예시(정상):

```json
{
  "status": "ready",
  "limits": {
    "max_concurrent_decode": 64,
    "max_concurrent_infer": 1,
    "batch_size": 64
  },
  "grpc": {
    "enabled": true,
    "host": "0.0.0.0",
    "port": 50051
  }
}
```

`ready` 대기용 예시:

```bash
until [ "$(curl -s http://localhost:8907/health | jq -r .status)" = "ready" ]; do
  echo "waiting for model load..."
  sleep 5
done
curl -s http://localhost:8907/health | jq .
```

3. 파일 로그 확인

```bash
ls -al logs
```

기대 결과:
- `logs/app.log` 파일이 존재해야 함

4. 컨테이너 로그 확인

```bash
docker logs qwen3-asr-offline --tail 100
```

기대 로그 키워드:
- `Application startup complete.`
- `gRPC server listening on 0.0.0.0:50051`
- `Server is ready to accept requests`

비정상 판단 기준:
- `status`가 장시간 `loading_models`에서 바뀌지 않음
- `docker logs`에 Python traceback 또는 `Model load failure`가 반복됨

### 5) 트러블슈팅 (운영 중 자주 발생)

포트 충돌:
- 증상: `Bind for 0.0.0.0:8907 failed: port is already allocated`
- 원인: 기존 서버(예: `qwen3-asr-fast`)가 같은 포트 사용 중
- 조치:

```bash
docker ps --format 'table {{.Names}}\t{{.Ports}}'
docker stop qwen3-asr-fast
make up-offline
```

모델 경로 누락:
- 증상: `Missing ASR model dir: ...`
- 조치:

```bash
ls -al models
ls -al models/1.7b-v2-cp47728
make up-offline
```

## 모델 교체 방법 (재빌드 없음)

예: 새 모델 디렉토리가 `/data/asr-models/1.7b-v2-new` 인 경우

```bash
make down-offline
make up-offline OFFLINE_ASR_MODEL_DIR=1.7b-v2-new
```

Aligner도 바꾸려면:

```bash
make up-offline \
  OFFLINE_ASR_MODEL_DIR=1.7b-v2-new \
  OFFLINE_ALIGNER_MODEL_DIR=0.6b-v3-new
```

모델 디렉토리만 교체하면 되며, 오프라인 이미지 재빌드는 필요 없습니다.

## 주요 설정 파일

- 온라인/compose: `config/runtime.json`
- 오프라인/docker run: `config/offline.runtime.json`
- 기본값: `config/default.json`

주요 키:
- `models.asr_model_path`
- `models.aligner_model_path`
- `models.local_files_only`
- `models.force_offline`
- `grpc.batch_wait_ms`
- `grpc.infer_batching_mode` (`dynamic` / `inflight`)

## 환경 변수 오버라이드

접두사 방식:
- `APP_CFG__SECTION__KEY=value`

예시:

```bash
APP_CFG__SERVER__PORT=8000
APP_CFG__MODELS__ENABLE_ALIGNER_MODEL=false
APP_CFG__MODELS__ASR_MODEL_PATH=/opt/models/1.7b-v2-cp47728
```

## 엔드포인트

- HTTP Transcribe: `POST /transcribe`
- WebSocket Streaming: `WS /transcribe-streaming`
- Health: `GET /health`
- gRPC: `Infer`, `BatchInfer`, `HealthCheck`

gRPC `InferRequest.audio`는 `float32 PCM bytes`(16kHz mono) 기준입니다.

## 테스트/벤치 명령

```bash
make test
make test-streaming
make test-grpc
make benchmark-batch
make benchmark-streaming
make benchmark-grpc-batch
```

예시 (1000개 오디오 gRPC 배치 벤치):

```bash
make benchmark-grpc-batch \
  GRPC_BENCH_AUDIO_DIR=/data/8_Daejeon_sobang/Training/audio \
  GRPC_BENCH_SAMPLES=1000 \
  GRPC_BENCH_BATCH=16
```

## 운영 참고

- Docker 로그 제한은 `docker/docker-compose.yml`에서 `max-size=10m`, `max-file=3` 적용
- 애플리케이션 파일 로그는 `logs/app.log`로 로테이션
- 권한 문제 시:

```bash
make fix-logs-perms
```
