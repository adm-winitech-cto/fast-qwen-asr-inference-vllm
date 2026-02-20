from __future__ import annotations

import logging

import grpc

from fast_qwen_asr.api.grpc_service import STTServiceServicer
from fast_qwen_asr.config.app_config import AppConfig
from fast_qwen_asr.core.runtime_state import RuntimeState
import stt_grpc_pb2_grpc

logger = logging.getLogger(__name__)


async def start_grpc_server(
    *,
    runtime_state: RuntimeState,
    config: AppConfig,
    stream_sample_rate: int,
) -> grpc.aio.Server:
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ]
    )
    servicer = STTServiceServicer(
        runtime_state=runtime_state,
        config=config,
        stream_sample_rate=stream_sample_rate,
    )
    stt_grpc_pb2_grpc.add_STTServiceServicer_to_server(servicer, server)
    setattr(server, "_stt_servicer", servicer)
    listen_addr = f"{config.grpc.host}:{config.grpc.port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("gRPC server listening on %s", listen_addr)
    return server
