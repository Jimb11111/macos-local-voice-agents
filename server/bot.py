import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict

# Add local pipecat to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipecat", "src"))

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v2 import LocalSmartTurnAnalyzerV2
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
# from kokoro_tts import KokoroTTSService
from kokoro_tts_isolated import KokoroTTSIsolated
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.transports.base_transport import TransportParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.processors.aggregators.llm_response import LLMUserAggregatorParams


load_dotenv(override=True)

pcs_map: Dict[str, SmallWebRTCConnection] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    logger.info("Shutting down, cleaning up connections...")
    coros = [pc.disconnect() for pc in pcs_map.values()]
    try:
        await asyncio.gather(*coros, return_exceptions=True)
    except Exception as e:
        logger.warning(f"Error during connection cleanup: {e}")
    finally:
        pcs_map.clear()
        logger.info("Cleanup completed")

app = FastAPI(lifespan=lifespan)

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]


SYSTEM_INSTRUCTION = """
"You are Pipecat, a cynical, angry chatbot.

Your input is text transcribed in realtime from the user's voice. There may be transcription errors. Adjust your responses automatically to account for these errors.

Your output will be converted to audio so don't include special characters in your answers and do not use any markdown or special formatting.

Respond to what the user said in a creative and helpful way. Keep your responses brief unless you are explicitly asked for long or detailed responses. Normally you should use one or two sentences at most. Keep each sentence short. Prefer simple sentences. Try not to use long sentences with multiple comma clauses.

Start the conversation by saying, "Hello, I'm Pipecat!" Then stop and wait for the user.
"""


async def run_bot(webrtc_connection):
    logger.info("Starting bot with WebRTC connection")
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV2(
                smart_turn_model_path="",  # Download from HuggingFace
                params=SmartTurnParams(),
            ),
        ),
    )
    logger.info(f"Audio settings - IN: True, OUT: True, Sample Rate: 24000")

    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4)

    # tts = KokoroTTSService(model="prince-canuma/Kokoro-82M", voice="af_heart", sample_rate=24000)
    # Process-isolated version to avoid Metal assertion failures (now refactored to use standalone worker)
    tts = KokoroTTSIsolated(model="prince-canuma/Kokoro-82M", voice="af_heart", sample_rate=24000)

    llm = OpenAILLMService(
        api_key="dummyKey",
        model="gemma-3-12b-it-qat",  # Medium-sized model. Uses ~8.5GB of RAM.
        # model="mlx-community/Qwen3-235B-A22B-Instruct-2507-3bit-DWQ", # Large model. Uses ~110GB of RAM!
        base_url="http://127.0.0.1:1234/v1",
        max_tokens=4096,
        model_kwargs={"max_context": 131072},  # Increase context length to 128k tokens
    )

    context = OpenAILLMContext(
        [
            {
                "role": "user",
                "content": SYSTEM_INSTRUCTION,
            }
        ],
    )
    context_aggregator = llm.create_context_aggregator(
        context,
        # Whisper local service isn't streaming, so it delivers the full text all at
        # once, after the UserStoppedSpeaking frame. Set aggregation_timeout to a
        # slightly higher value to handle user interruptions more gracefully.
        user_params=LLMUserAggregatorParams(aggregation_timeout=0.2),
    )

    #
    # RTVI events for Pipecat client UI
    #
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            rtvi,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    logger.info("Pipeline created with: Input -> STT -> RTVI -> Context -> LLM -> TTS -> Output")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready, starting conversation")
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info(f"Participant joined: {participant}")
        await transport.capture_participant_transcription(participant["id"])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info(f"Participant left: {participant} - Reason: {reason}")
        try:
            await task.cancel()
        except Exception as e:
            logger.warning(f"Error during task cancellation: {e}")

    runner = PipelineRunner(handle_sigint=False)
    logger.info("Starting pipeline runner")
    
    await runner.run(task)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Run example function with SmallWebRTC transport arguments.
        background_tasks.add_task(run_bot, pipecat_connection)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipecat Bot Runner")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
