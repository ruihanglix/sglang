"""
OpenAI-compatible /v1/chat/completions endpoint for HunyuanImage-3.0.
Supports text generation for prompt enhancement and TI2T tasks.
Routes requests through the scheduler to the pipeline.
"""

import base64
import time
import uuid
from io import BytesIO
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

router = APIRouter()


class ChatMessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ChatMessageContent]]


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    seed: Optional[int] = None
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage = Usage()


def _parse_messages(messages: List[ChatMessage]):
    """Extract prompt text and images from OpenAI-style messages."""
    prompt_parts = []
    images = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                prompt_parts.append(msg.content)
            continue

        if msg.role != "user":
            continue

        if isinstance(msg.content, str):
            prompt_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if item.type == "text" and item.text:
                    prompt_parts.append(item.text)
                elif item.type == "image_url" and item.image_url:
                    url = item.image_url.get("url", "")
                    images.append(url)

    prompt = "\n".join(prompt_parts)
    return prompt, images


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports text generation for prompt enhancement and TI2T tasks.
    """
    if body.stream:
        raise HTTPException(
            status_code=400, detail="Streaming is not supported for this model."
        )

    server_args = get_global_server_args()
    prompt, image_urls = _parse_messages(body.messages)

    if not prompt:
        raise HTTPException(status_code=400, detail="No text content in messages.")

    rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Create sampling params for text generation
    sp = SamplingParams()
    sp.prompt = prompt
    sp.request_id = rid
    if body.seed is not None:
        sp.seed = body.seed

    # Create a request and mark it for text generation
    req = prepare_request(server_args, sampling_params=sp)
    req.text_gen_mode = True
    req.text_gen_max_tokens = body.max_tokens
    req.text_gen_image_urls = image_urls

    try:
        response = await async_scheduler_client.forward(req)

        generated_text = ""
        if response.output is not None:
            if isinstance(response.output, str):
                generated_text = response.output
            elif isinstance(response.output, list) and len(response.output) > 0:
                generated_text = str(response.output[0])
            else:
                generated_text = str(response.output)

    except Exception as e:
        logger.error("Text generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    result = ChatCompletionResponse(
        id=rid,
        created=int(time.time()),
        model=body.model or server_args.model_path,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(generated_text.split()),
            total_tokens=len(prompt.split()) + len(generated_text.split()),
        ),
    )

    return ORJSONResponse(content=result.model_dump())
