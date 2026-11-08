"""A streamed chat completion is read to the end of its HTTP body.

`_stream_response` / `_stream_response_async` stop processing at the usage chunk (at
the finish chunk when not streaming), before the stream's `[DONE]` event and the end
of its body. Left unread, the response is only closed when the stream is garbage-
collected, which on HTTP/1.1 drops its connection: every turn paid a new TCP + TLS
handshake. The unit tests check that what the loop leaves is read on; the last test
runs the real OpenAI SDK against a local HTTP/1.1 server and counts connections.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import openai
import pytest

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

# what a chat-completions stream sends with stream_options.include_usage, plus one
# event after the usage chunk - standing in for the `[DONE]` the SDK consumes itself
_EVENTS: list[dict[str, Any]] = [
    {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]},
    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    {
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    },
    {"choices": []},
]


class _Events(Iterator[dict[str, Any]]):
    """Synthetic stream events that record how far they were read."""

    def __init__(self, events: list[dict[str, Any]], fail_at: int = -1) -> None:
        self._events = events
        self._fail_at = fail_at
        self.read = 0

    def __iter__(self) -> "_Events":
        return self

    def __next__(self) -> dict[str, Any]:
        if self.read == self._fail_at:
            raise ConnectionError("peer closed connection")
        if self.read >= len(self._events):
            raise StopIteration
        self.read += 1
        return self._events[self.read - 1]

    @property
    def exhausted(self) -> bool:
        return self.read == len(self._events)


class _AsyncEvents(AsyncIterator[dict[str, Any]]):
    """The async twin of `_Events`."""

    def __init__(self, events: list[dict[str, Any]], fail_at: int = -1) -> None:
        self._sync = _Events(events, fail_at)

    def __aiter__(self) -> "_AsyncEvents":
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._sync)
        except StopIteration:
            raise StopAsyncIteration from None

    @property
    def exhausted(self) -> bool:
        return self._sync.exhausted


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
def test_sync_stream_is_read_to_the_end(stream: bool) -> None:
    model = OpenAIGPT(OpenAIGPTConfig(stream=stream))
    events = _Events(_EVENTS)

    response, _ = model._stream_response(events, chat=True)

    assert response.message == "hi"
    assert events.exhausted


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "no-stream"])
async def test_async_stream_is_read_to_the_end(stream: bool) -> None:
    model = OpenAIGPT(OpenAIGPTConfig(stream=stream))
    events = _AsyncEvents(_EVENTS)

    response, _ = await model._stream_response_async(events, chat=True)

    assert response.message == "hi"
    assert events.exhausted


def test_sync_failure_after_the_answer_keeps_the_answer() -> None:
    """The connection dropping while the rest is read costs the connection only."""
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))

    response, _ = model._stream_response(_Events(_EVENTS, fail_at=3), chat=True)

    assert response.message == "hi"
    assert response.usage is not None and response.usage.completion_tokens == 1


@pytest.mark.asyncio
async def test_async_failure_after_the_answer_keeps_the_answer() -> None:
    model = OpenAIGPT(OpenAIGPTConfig(stream=True))

    response, _ = await model._stream_response_async(
        _AsyncEvents(_EVENTS, fail_at=3), chat=True
    )

    assert response.message == "hi"
    assert response.usage is not None and response.usage.completion_tokens == 1


def _chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


class _Server:
    """Streams a chat completion per request, as the API does - `[DONE]` first, the
    end of the chunked body a moment later - on keep-alive connections it counts."""

    def __init__(self) -> None:
        self.connections = 0

    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.connections += 1
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode().lower()
                length = head.split("content-length:")[1].split("\r\n")[0]
                await reader.readexactly(int(length))
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                base = {"id": "x", "object": "chat.completion.chunk", "model": "m"}
                for event in _EVENTS[:3]:
                    body = json.dumps({**base, "created": 0, **event})
                    writer.write(_chunk(f"data: {body}\n\n".encode()))
                writer.write(_chunk(b"data: [DONE]\n\n"))
                await writer.drain()
                await asyncio.sleep(0.05)
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    tuple(int(p) for p in openai.__version__.split(".")[:2]) < (3, 30),
    reason="openai<3.30 closes a stream's response at [DONE] without reading on",
)
async def test_streamed_turns_reuse_one_connection() -> None:
    server = _Server()
    listener = await asyncio.start_server(server.handle, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        model = OpenAIGPT(
            OpenAIGPTConfig(
                chat_model="m",
                api_base=f"http://127.0.0.1:{port}/v1",
                api_key="x",
                stream=True,
                use_cached_client=False,
                cache_config=None,
            )
        )
        for _ in range(3):
            response = await model.achat("hi", max_tokens=10)
            assert response.message == "hi"
        assert server.connections == 1
    finally:
        listener.close()
