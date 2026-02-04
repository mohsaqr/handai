"""
Mock LLM Client for Testing
Simulates OpenAI API responses without making real API calls
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio


@dataclass
class MockChoice:
    """Mock response choice"""
    message: 'MockMessage'
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class MockMessage:
    """Mock message"""
    content: str
    role: str = "assistant"


@dataclass
class MockResponse:
    """Mock API response"""
    choices: List[MockChoice]
    model: str = "mock-model"
    id: str = "mock-id"

    @classmethod
    def create(cls, content: str, model: str = "mock-model"):
        return cls(
            choices=[MockChoice(message=MockMessage(content=content))],
            model=model
        )


class MockCompletions:
    """Mock completions endpoint"""

    def __init__(self, client: 'MockLLMClient'):
        self.client = client

    async def create(self, **kwargs) -> MockResponse:
        """Mock create completion"""
        self.client.call_count += 1
        self.client.last_kwargs = kwargs

        # Simulate delay
        if self.client.delay > 0:
            await asyncio.sleep(self.client.delay)

        # Raise error if configured
        if self.client.error_to_raise:
            raise self.client.error_to_raise

        # Return configured response
        if self.client.responses:
            response = self.client.responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

        return MockResponse.create(self.client.default_response)


class MockChat:
    """Mock chat endpoint"""

    def __init__(self, client: 'MockLLMClient'):
        self.completions = MockCompletions(client)


class MockLLMClient:
    """
    Mock AsyncOpenAI client for testing.

    Usage:
        client = MockLLMClient(default_response='{"result": "test"}')
        response = await client.chat.completions.create(...)
    """

    def __init__(
        self,
        default_response: str = "Mock response",
        responses: List[Any] = None,
        error_to_raise: Exception = None,
        delay: float = 0
    ):
        self.default_response = default_response
        self.responses = responses or []
        self.error_to_raise = error_to_raise
        self.delay = delay
        self.call_count = 0
        self.last_kwargs: Dict[str, Any] = {}
        self.chat = MockChat(self)

    def reset(self):
        """Reset call tracking"""
        self.call_count = 0
        self.last_kwargs = {}

    def set_responses(self, responses: List[Any]):
        """Set sequence of responses (can include exceptions)"""
        self.responses = responses

    def set_error(self, error: Exception):
        """Set error to raise on next call"""
        self.error_to_raise = error

    def clear_error(self):
        """Clear error"""
        self.error_to_raise = None


# Predefined error scenarios
class MockAPIError(Exception):
    """Base mock API error"""
    pass


class MockRateLimitError(MockAPIError):
    """Mock rate limit error (429)"""
    def __init__(self):
        super().__init__("Rate limit exceeded. Error code: 429")


class MockAuthError(MockAPIError):
    """Mock authentication error (401)"""
    def __init__(self):
        super().__init__("Invalid API key. Error code: 401 Unauthorized")


class MockTimeoutError(MockAPIError):
    """Mock timeout error"""
    def __init__(self):
        super().__init__("Request timed out after 30 seconds")


class MockConnectionError(MockAPIError):
    """Mock connection error"""
    def __init__(self):
        super().__init__("Connection refused - could not connect to server")


class MockModelNotFoundError(MockAPIError):
    """Mock model not found error (404)"""
    def __init__(self, model: str = "unknown-model"):
        super().__init__(f"Model '{model}' not found. Error code: 404")


class MockContentFilterError(MockAPIError):
    """Mock content filter error"""
    def __init__(self):
        super().__init__("Content blocked by safety filter")


class MockContextLengthError(MockAPIError):
    """Mock context length exceeded error"""
    def __init__(self):
        super().__init__("Context length exceeded. Maximum context length is 4096 tokens.")


class MockServerError(MockAPIError):
    """Mock server error (500)"""
    def __init__(self):
        super().__init__("Internal server error. Error code: 500")
