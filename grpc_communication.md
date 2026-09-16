# gRPC 통신 가이드

이 문서는 현재 서버 구현(`stt_grpc.proto`) 기준으로, STT gRPC를 실제 연동할 때 필요한 전송 포맷/호출 순서를 정리합니다.

## 1) 엔드포인트
- 서버 바인드: `0.0.0.0:50051`
- 클라이언트 접속 예시: `127.0.0.1:50051` (또는 서버 IP)
- 서비스: `stt.STTService`
- 메서드:
  - `Infer(InferRequest) -> InferResponse`
  - `BatchInfer(BatchInferRequest) -> BatchInferResponse`
  - `HealthCheck(HealthCheckRequest) -> HealthCheckResponse`

## 2) 핵심 전송 규칙 (중요)
- 오디오 필드(`bytes`)에는 **파일 원본 바이너리(mp3/wav)를 그대로 넣지 않고**,
  **16kHz / mono / float32 PCM raw bytes**를 넣어야 합니다.
- 즉, 클라이언트에서 디코딩/리샘플 후 `audio_float32.tobytes()` 형태로 전송합니다.

권장 ffmpeg 변환 예시:
```bash
ffmpeg -nostdin -v error -i input.wav -f f32le -ac 1 -ar 16000 pipe:1
```

## 3) Proto 필드 의미

### InferRequest
- `audio`: 단일 오디오의 float32 PCM bytes
- `lang`: 언어 코드 (`ko`, `en` 등)
- `pretext`: 선택 프롬프트(없으면 빈 문자열)
- `model_name`: 고정 요청 모델명 (기본: `asr`)

### BatchInferRequest
- `audios`: `repeated bytes` (항목별 float32 PCM bytes)
- `langs`: 언어 목록. 일반적으로 `["ko"]` 1개만 넣어 전체 배치에 적용
- `model_name`: 배치 전체에 적용할 고정 모델명 (`asr`)

### 응답
- `InferResponse`: `text`, `success`, `error`
- `BatchInferResponse.items[]`: 각 오디오별 `text`, `success`, `error`

## 4) 호출 순서 권장
1. `HealthCheck`로 준비 상태 확인 (`ready=true`, 모델 상태 확인)
2. 단건/저QPS는 `Infer` 사용  
   참고: 현재 서버는 `Infer` 요청도 내부 큐에서 `batch_wait_ms` 동안 모아
   in-flight micro-batching(동적 합배치)으로 처리합니다.  
3. 대량/고QPS는 `BatchInfer` 사용 (`repeated bytes`)  
   이유: RPC 호출 수 감소, 큐/스케줄링 오버헤드 감소
4. 실패 항목은 `items[i].success=false` / `error`로 개별 처리

## 5) Python 예시

### HealthCheck
```bash
python client-grpc.py --endpoint 127.0.0.1:50051 --health --health-model asr
```

### Infer (단건)
```bash
python client-grpc.py \
  --endpoint 127.0.0.1:50051 \
  --file files/reference.wav \
  --lang ko \
  --model-name asr \
  --batch-size 1 \
  --requests 1
```

### BatchInfer 벤치마크 (실제 repeated bytes)
```bash
make benchmark-grpc-batch \
  GRPC_BENCH_AUDIO_DIR=/data/8_Daejeon_sobang/Training/audio \
  GRPC_BENCH_SAMPLES=1000 \
  GRPC_BENCH_BATCH=16
```

## 6) 운영 튜닝 포인트
- `config/runtime.json`
  - `concurrency.max_concurrent_decode`
  - `concurrency.batch_size`
  - `grpc.batch_wait_ms`
  - `grpc.infer_batching_mode` (`dynamic` or `inflight`)
- 변경 후 컨테이너 재시작:
```bash
docker restart qwen3-asr-fast
```

## 7) 자주 발생하는 실수
- mp3/wav 파일 바이트를 그대로 `audio`에 넣는 경우
- 샘플레이트/채널이 16k mono가 아닌 상태로 전송
- `BatchInfer`인데 `audios`가 빈 배열인 경우
- `model_name`은 `config/runtime.json`의 `grpc.request_model_name`과 동일해야 함 (기본 `asr`)

## 8) InferRequest 속도 관련
- `InferRequest` 자체가 문제는 아닙니다.
- 느려지는 주요 원인은 모델 추론 시간, 서버 큐 대기, gRPC 전송량(큰 오디오)입니다.
- 현재 서버는 `Infer`도 내부 합배치를 사용하므로, 동시 요청이 많을수록 처리량이 크게 증가합니다.
- 단건 API를 유지하더라도 클라이언트에서 동시 요청(`--batch-size`)을 높여 파이프라인을 채우는 것이 유효합니다.
