"""
Handai LLM Client
Client creation and API calling utilities
"""

import asyncio
import time
from typing import Optional, List, Tuple, Any
import httpx
from openai import AsyncOpenAI

from .providers import (
    LLMProvider, PROVIDER_CONFIGS,
    is_reasoning_model, uses_completion_tokens, supports_json_mode
)
from errors import ErrorClassifier, ErrorInfo
from database import get_db, LogLevel
from config import (
    HTTP_MAX_CONNECTIONS, HTTP_MAX_KEEPALIVE,
    HTTP_KEEPALIVE_EXPIRY, HTTP_TIMEOUT, HTTP_CONNECT_TIMEOUT
)


def create_http_client() -> httpx.AsyncClient:
    """Create a configured HTTP client for LLM requests"""
    return httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=HTTP_MAX_CONNECTIONS,
            max_keepalive_connections=HTTP_MAX_KEEPALIVE,
            keepalive_expiry=HTTP_KEEPALIVE_EXPIRY
        ),
        timeout=httpx.Timeout(HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT)
    )


def get_client(provider: LLMProvider, api_key: str, base_url: str,
               http_client: Optional[httpx.AsyncClient] = None) -> AsyncOpenAI:
    """Create appropriate AsyncOpenAI client for the provider"""
    config = PROVIDER_CONFIGS[provider]
    effective_key = api_key if config.requires_api_key else "dummy"
    effective_url = base_url or config.base_url

    # OpenRouter requires additional headers
    if provider == LLMProvider.OPENROUTER:
        return AsyncOpenAI(
            api_key=effective_key,
            base_url=effective_url,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "https://handai.app",
                "X-Title": "Handai Data Transformer"
            }
        )

    # Other providers use the custom http_client if provided
    if http_client:
        return AsyncOpenAI(
            api_key=effective_key,
            base_url=effective_url,
            http_client=http_client,
            max_retries=0
        )

    return AsyncOpenAI(
        api_key=effective_key,
        base_url=effective_url,
        max_retries=0
    )


def fetch_local_models(base_url: str) -> List[str]:
    """Fetch available models from LM Studio or Ollama /v1/models endpoint"""
    try:
        response = httpx.get(f"{base_url}/models", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


def fetch_openrouter_models() -> List[str]:
    """Fetch available models from OpenRouter API"""
    try:
        response = httpx.get("https://openrouter.ai/api/v1/models", timeout=10.0)
        if response.status_code == 200:
            data = response.json()
            # Return top 50 models sorted by ID (OpenRouter returns them sorted by popularity)
            models = [m["id"] for m in data.get("data", [])[:50]]
            return models if models else []
    except Exception:
        pass
    # Fallback to default models if fetch fails
    return [
        "anthropic/claude-sonnet-4",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ]


async def call_llm_with_retry(
    client: AsyncOpenAI,
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    run_id: str = None,
    row_index: int = 0,
    max_retries: int = 3,
    db=None
) -> Tuple[Optional[str], float, Optional[ErrorInfo], int]:
    """
    Call LLM with error handling and auto-retry for empty results.

    Returns:
        Tuple of (output, duration, error_info, attempt_count)
    """
    if db is None:
        db = get_db()

    attempt = 0
    last_error = None

    while attempt <= max_retries:
        start = time.time()
        try:
            # Build request kwargs
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
            }

            # Add temperature only for models that support it (not o1/o3/gpt-5)
            if not is_reasoning_model(model):
                kwargs["temperature"] = temperature

            # Use appropriate token parameter
            if uses_completion_tokens(model):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

            # Add JSON mode if supported
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content
            duration = time.time() - start

            # Check if result is empty/NA - auto-retry if so
            # Skip retry for reasoning models as they may return empty during reasoning
            if not is_reasoning_model(model) and ErrorClassifier.is_empty_result(output) and attempt < max_retries:
                if run_id:
                    db.log(LogLevel.WARNING, f"Empty result for row {row_index}, retrying...",
                           {"attempt": attempt, "output": output[:100] if output else None}, run_id=run_id)
                attempt += 1
                await asyncio.sleep(1)  # Brief delay before retry
                continue

            return output, duration, None, attempt

        except Exception as e:
            duration = time.time() - start
            error_info = ErrorClassifier.classify(e)

            # Log the error
            if run_id:
                db.log(LogLevel.ERROR, f"Error on row {row_index}: {error_info.message}",
                       {"error_type": error_info.error_type.value, "original": error_info.original_error,
                        "attempt": attempt}, run_id=run_id)

            # Check if we should retry
            should_retry, delay = ErrorClassifier.should_auto_retry(error_info, attempt, max_retries)

            if should_retry:
                if run_id:
                    db.log(LogLevel.INFO, f"Retrying row {row_index} after {delay}s delay",
                           {"attempt": attempt + 1}, run_id=run_id)
                await asyncio.sleep(delay)
                attempt += 1
                last_error = error_info
                continue
            else:
                return None, duration, error_info, attempt

    # Max retries exceeded
    return None, 0, last_error, attempt


async def call_llm_simple(
    client: AsyncOpenAI,
    system_prompt: str,
    user_content: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    json_mode: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    Simple LLM call without retry logic.

    Returns:
        Tuple of (output, error_message)
    """
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
        }

        if not is_reasoning_model(model):
            kwargs["temperature"] = temperature

        if uses_completion_tokens(model):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, None

    except Exception as e:
        error_info = ErrorClassifier.classify(e)
        return None, error_info.message
