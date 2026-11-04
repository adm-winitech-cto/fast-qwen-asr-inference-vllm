IMAGE              ?= qwenllm/qwen3-asr:cu128-sbsa
CONTAINER          ?= qwen3-asr-fast
PORT               ?= 8907
GRPC_PORT          ?= 50051
COMPOSE_FILE       ?= docker/docker-compose.yml
OFFLINE_IMAGE      ?= qwenllm/qwen3-asr:offline-test
OFFLINE_CONTAINER  ?= qwen3-asr-offline
OFFLINE_TAR        ?= dist/qwen3-asr-offline.tar
OFFLINE_CONFIG     ?= /opt/fast-qwen-asr/config/offline.runtime.json
OFFLINE_MODELS_DIR ?= $(PWD)/models
OFFLINE_MODELS_MOUNT ?= /opt/models
OFFLINE_ASR_MODEL_DIR ?= 1.7b-v2-cp47728
OFFLINE_ALIGNER_MODEL_DIR ?= 0.6b-v3-cp22426
OFFLINE_GPU_UTIL   ?= 0.2
FILES_DIR          ?= files
TEST_URL_LIST      ?= https://raw.githubusercontent.com/kyr0/asr-sample-files/refs/heads/main/index.m3u
CONFIG_FILE        ?= $(PWD)/config/runtime.json
CONTAINER_CONFIG_FILE ?= /data/shared/Qwen3-ASR/config/runtime.json
ENV_FILE           ?= config/docker.env
GRPC_BENCH_AUDIO_DIR ?= /data/8_Daejeon_sobang/Training/audio
GRPC_BENCH_SAMPLES ?= 1000
GRPC_BENCH_BATCH ?= 16
HOST_UID ?= $(shell id -u)
HOST_GID ?= $(shell id -g)

.PHONY: build up down logs health test setup-local test-streaming test-batch test-aligner test-grpc benchmark-streaming benchmark-batch benchmark-grpc-batch prepare-logs fix-logs-perms build-offline save-offline load-offline up-offline down-offline logs-offline

build:
	docker build --rm -t $(IMAGE) -f docker/Dockerfile .

build-offline:
	docker build --network none --rm --build-arg BASE_IMAGE=$(IMAGE) -t $(OFFLINE_IMAGE) -f docker/Dockerfile.offline .

save-offline:
	@mkdir -p $(dir $(OFFLINE_TAR))
	docker save $(OFFLINE_IMAGE) -o $(OFFLINE_TAR)
	@echo "Saved: $(OFFLINE_TAR)"

load-offline:
	docker load -i $(OFFLINE_TAR)

up-offline:
	@$(MAKE) prepare-logs
	@test -d "$(OFFLINE_MODELS_DIR)/$(OFFLINE_ASR_MODEL_DIR)" || (echo "Missing ASR model dir: $(OFFLINE_MODELS_DIR)/$(OFFLINE_ASR_MODEL_DIR)"; exit 1)
	-docker rm -f $(OFFLINE_CONTAINER)
	docker run -d --name $(OFFLINE_CONTAINER) \
	  --gpus all --ipc=host \
	  -p $(PORT):8000 \
	  -p $(GRPC_PORT):50051 \
	  -e APP_CONFIG_PATH=$(OFFLINE_CONFIG) \
	  -e APP_CFG__MODELS__ASR_MODEL_PATH=$(OFFLINE_MODELS_MOUNT)/$(OFFLINE_ASR_MODEL_DIR) \
	  -e APP_CFG__MODELS__ALIGNER_MODEL_PATH=$(OFFLINE_MODELS_MOUNT)/$(OFFLINE_ALIGNER_MODEL_DIR) \
	  -e APP_CFG__MODELS__GPU_MEMORY_UTILIZATION=$(OFFLINE_GPU_UTIL) \
	  -e HF_HUB_OFFLINE=1 \
	  -e TRANSFORMERS_OFFLINE=1 \
	  -e HF_DATASETS_OFFLINE=1 \
	  --log-driver json-file \
	  --log-opt max-size=10m \
	  --log-opt max-file=3 \
	  -v $(PWD)/logs:/opt/fast-qwen-asr/logs \
	  -v $(OFFLINE_MODELS_DIR):$(OFFLINE_MODELS_MOUNT):ro \
	  $(OFFLINE_IMAGE)
	@$(MAKE) logs-offline

down-offline:
	-docker rm -f $(OFFLINE_CONTAINER)

logs-offline:
	docker logs -f $(OFFLINE_CONTAINER)

up:
	@echo "Starting service with compose file: $(COMPOSE_FILE)"
	@$(MAKE) prepare-logs
	mkdir -p $(FILES_DIR)
	docker compose -f $(COMPOSE_FILE) up -d --build
	@$(MAKE) logs

down:
	-docker compose -f $(COMPOSE_FILE) down

logs:
	docker compose -f $(COMPOSE_FILE) logs -f asr-server

prepare-logs:
	@mkdir -p logs
	@chown -R $(HOST_UID):$(HOST_GID) logs 2>/dev/null || true
	@chmod -R ug+rwX logs 2>/dev/null || true
	@docker run --rm -v $(PWD)/logs:/logs alpine:3.20 sh -c "chown -R $(HOST_UID):$(HOST_GID) /logs && chmod -R 775 /logs" >/dev/null 2>&1 || true

fix-logs-perms:
	@$(MAKE) prepare-logs

health:
	@echo "Checking health..."
	@curl -s http://localhost:$(PORT)/health | jq .

test:
	@echo "Running verification test..."
	@mkdir -p $(FILES_DIR)
	@# Download samples if missing
	@for ext in m4a mp3 wav; do \
		if [ ! -f "$(FILES_DIR)/reference.$$ext" ]; then \
			echo "Downloading reference.$$ext..."; \
			wget -q -O $(FILES_DIR)/reference.$$ext https://raw.githubusercontent.com/kyr0/asr-sample-files/refs/heads/main/reference.$$ext; \
		fi \
	done
	
	@pass=0; fail=0; \
	for ext in m4a mp3 wav; do \
		file="$(FILES_DIR)/reference.$$ext"; \
		echo "Testing $$file"; \
		resp=$$(curl -s -X POST -F "files=@$$file" "http://localhost:$(PORT)/transcribe?language=de"); \
		echo "Response: $$resp"; \
		if echo "$$resp" | grep -iq "referenz"; then \
			echo "PASS: 'referenz' found."; \
			pass=$$((pass+1)); \
		else \
			echo "FAIL: 'referenz' not found."; \
			fail=$$((fail+1)); \
		fi; \
	done; \
	echo "Results: $$pass PASSED, $$fail FAILED"; \
	if [ $$fail -gt 0 ]; then exit 1; fi

setup-local:
	@echo "Setting up local Python environment..."
	@if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	@. .venv/bin/activate && pip install -r requirements.txt
	@echo "Setting up local Node.js environment..."
	@npm install --silent

test-streaming:
	@echo "Running Python Streaming Test..."
	@. .venv/bin/activate && python client-streaming.py -e ws://127.0.0.1:$(PORT)/transcribe-streaming -f $(FILES_DIR)/reference.pcm
	@echo "\nRunning Node.js Streaming Test..."
	@node client-streaming.js -e ws://127.0.0.1:$(PORT)/transcribe-streaming -f $(FILES_DIR)/reference.pcm

test-batch:
	@echo "Running Batch Verification Test..."
	@# Send multiple files (reusing the same file 3 times for batching test)
	@curl -s -X POST "http://127.0.0.1:$(PORT)/transcribe?language=de" \
		-F "files=@$(FILES_DIR)/reference.m4a" \
		-F "files=@$(FILES_DIR)/reference.mp3" \
		-F "files=@$(FILES_DIR)/reference.wav" | jq .

test-aligner:
	@echo "Running Forced Alignment Verification Test..."
	@curl -s -X POST "http://127.0.0.1:$(PORT)/transcribe?language=de&forced_alignment=true" \
		-F "files=@$(FILES_DIR)/reference.wav" | jq .

test-grpc:
	@echo "Running gRPC HealthCheck..."
	@. .venv/bin/activate && python client-grpc.py --endpoint 127.0.0.1:$(GRPC_PORT) --health --health-model asr
	@echo "Running gRPC Infer..."
	@. .venv/bin/activate && python client-grpc.py --endpoint 127.0.0.1:$(GRPC_PORT) --file $(FILES_DIR)/reference.wav --lang ko --batch-size 2 --requests 2

benchmark-streaming:
	@echo "Running Streaming Benchmark (Concurrency: 4, Requests: 20)..."
	@. .venv/bin/activate && python benchmark.py --mode streaming --url ws://127.0.0.1:$(PORT)/transcribe-streaming --file $(FILES_DIR)/reference.pcm --clients 4 --requests 20

benchmark-batch:
	@echo "Running Batch Benchmark (Concurrency: 4, Requests: 20)..."
	@. .venv/bin/activate && python benchmark.py --mode batch --url http://127.0.0.1:$(PORT)/transcribe --file $(FILES_DIR)/reference.wav --clients 4 --requests 20

benchmark-grpc-batch:
	@echo "Running gRPC BatchInfer Benchmark..."
	@echo "  AUDIO_DIR: $(GRPC_BENCH_AUDIO_DIR)"
	@echo "  SAMPLES:   $(GRPC_BENCH_SAMPLES)"
	@echo "  BATCH:     $(GRPC_BENCH_BATCH)"
	@docker run --rm --network host \
		-v $(PWD):/work \
		-v $(GRPC_BENCH_AUDIO_DIR):/bench_audio:ro \
		$(IMAGE) \
		python3 /work/grpc_batch_benchmark.py \
			--endpoint 127.0.0.1:$(GRPC_PORT) \
			--audio-dir /bench_audio \
			--sample-size $(GRPC_BENCH_SAMPLES) \
			--request-batch-size $(GRPC_BENCH_BATCH) \
			--model-name 1.7b-v2-cp47728
