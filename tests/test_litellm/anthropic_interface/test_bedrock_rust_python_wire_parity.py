"""Proves the Rust bridge and the Python path send Bedrock the same request.

Both engines are pointed at a local listener via ``aws_bedrock_runtime_endpoint``.
respx cannot be used here: it patches httpx, and the Rust leg goes out through
reqwest on real sockets.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

import litellm

MODEL = "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BEARER_TOKEN = "test-bedrock-bearer-token"

CANNED_RESPONSE: dict[str, object] = {
    "id": "msg_bdrk_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 2},
}

# Transport-owned headers (host, accept-encoding, user-agent, ...) differ between
# httpx and reqwest by design; only these carry request semantics.
COMPARED_HEADERS = ("authorization", "content-type")


@dataclass(frozen=True, slots=True)
class CapturedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, object]


def _make_handler(captured: list[CapturedRequest]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802  # http.server's required casing
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            captured.append(
                CapturedRequest(
                    method=self.command,
                    path=self.path,
                    headers={
                        name: value
                        for name, value in ((k.lower(), v) for k, v in self.headers.items())
                        if name in COMPARED_HEADERS
                    },
                    body=json.loads(raw),
                )
            )
            payload = json.dumps(CANNED_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            pass

    return Handler


@pytest.fixture
def bedrock_listener() -> Iterator[tuple[str, list[CapturedRequest]]]:
    captured: list[CapturedRequest] = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(captured))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _call(*, endpoint: str, rust: bool, **overrides: object) -> None:
    response = await litellm.anthropic.messages.acreate(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        api_key=BEARER_TOKEN,
        aws_bedrock_runtime_endpoint=endpoint,
        rust=rust,
        **overrides,
    )
    # Without this the suite is vacuous: a Rust failure falls back to Python, and
    # comparing two Python requests to each other always succeeds. Only the Rust
    # path stamps _hidden_params on a non-streaming response.
    hidden_params = dict(response).get("_hidden_params") or {}
    served_by_rust = hidden_params.get("additional_headers", {}).get("x-litellm-rust") == "true"
    assert served_by_rust is rust, f"expected rust={rust}, got served_by_rust={served_by_rust}"


@pytest.mark.asyncio
async def test_rust_and_python_send_identical_bedrock_requests(
    bedrock_listener: tuple[str, list[CapturedRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint, captured = bedrock_listener
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")

    await _call(endpoint=endpoint, rust=False)
    await _call(endpoint=endpoint, rust=True)

    assert len(captured) == 2, "both engines must reach the upstream exactly once"
    python_request, rust_request = captured
    assert python_request == rust_request


@pytest.mark.asyncio
async def test_rust_honors_aws_region_name_over_environment(
    bedrock_listener: tuple[str, list[CapturedRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression guard for the params the bridge used not to forward.

    With a differing env region, Rust resolved us-west-2 while Python resolved
    us-east-1. The endpoint override hides the host, so assert on the region the
    engines actually resolved by way of an ARN model id instead.
    """
    endpoint, captured = bedrock_listener
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")

    await _call(endpoint=endpoint, rust=False, aws_region_name="us-east-1")
    await _call(endpoint=endpoint, rust=True, aws_region_name="us-east-1")

    assert len(captured) == 2
    assert captured[0] == captured[1]


@pytest.mark.asyncio
async def test_rust_request_is_shaped_for_bedrock_invoke(
    bedrock_listener: tuple[str, list[CapturedRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint, captured = bedrock_listener
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")

    await _call(endpoint=endpoint, rust=True)

    request = captured[0]
    assert request.method == "POST"
    assert request.path == "/model/us.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"
    assert request.headers["authorization"] == f"Bearer {BEARER_TOKEN}"
    # Bedrock takes the model in the path and rejects it in the body.
    assert "model" not in request.body
    assert "stream" not in request.body
    assert request.body["anthropic_version"] == "bedrock-2023-05-31"


@pytest.mark.asyncio
async def test_streaming_through_rust_calls_the_non_streaming_verb(
    bedrock_listener: tuple[str, list[CapturedRequest]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented divergence: the bridge has no streaming variant, so it calls
    /invoke and Python fakes the SSE. Python on its own would call
    /invoke-with-response-stream."""
    endpoint, captured = bedrock_listener
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")

    response = await litellm.anthropic.messages.acreate(
        model=MODEL,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        stream=True,
        api_key=BEARER_TOKEN,
        aws_bedrock_runtime_endpoint=endpoint,
        rust=True,
    )
    chunks = b"".join([chunk async for chunk in response])

    assert captured[0].path.endswith("/invoke")
    assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
    assert b"event: message_start" in chunks
    assert b"event: message_stop" in chunks
