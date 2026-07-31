"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Awaitable, Final, Protocol, TypedDict, Union, cast

import httpx
import inspect
from litellm._logging import _is_debugging_on

from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustMessages(Protocol):
    # `optional_params` and `debug` default, mirroring the native signature, so a
    # bridge predating either can still be called.
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
        optional_params: dict[str, object] = ...,
        debug: bool = ...,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAmessages(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
        optional_params: dict[str, object] = ...,
        debug: bool = ...,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustMessagesState:
    messages: RustMessages | None = None
    amessages: RustAmessages | None = None


_STATE: Final[_RustMessagesState] = _RustMessagesState()


def set_rust_messages(
    *,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
) -> None:
    if not isinstance(messages, _Unset):
        _STATE.messages = messages
    if not isinstance(amessages, _Unset):
        _STATE.amessages = amessages


def load_rust_messages() -> RustMessages | None:
    if _STATE.messages is not None:
        return _STATE.messages
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustMessages, getattr(native_bridge, "messages", None))


def load_rust_amessages() -> RustAmessages | None:
    if _STATE.amessages is not None:
        return _STATE.amessages
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustAmessages, getattr(native_bridge, "amessages", None))


class _CommonBridgeKwargs(TypedDict):
    model: str
    body: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    timeout_seconds: float | None


def _common_kwargs(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
) -> _CommonBridgeKwargs:
    return _CommonBridgeKwargs(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )


def _declares(bridge: Callable[..., object], parameter: str) -> bool:
    """A native extension left over from an older build can be missing the newer
    arguments; passing them anyway raises a TypeError the caller swallows into a
    silent Python fallback. The signatures only ever grew, so a bridge that knows
    `optional_params` also knows `debug`."""
    return parameter in inspect.signature(bridge).parameters


def messages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: Union[float, httpx.Timeout] | None,
    optional_params: dict[str, object] | None = None,
) -> dict[str, object] | None:
    rust_messages = load_rust_messages()
    if rust_messages is None:
        return None
    common = _common_kwargs(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout=timeout,
    )
    if _declares(rust_messages, "optional_params"):
        return rust_messages(
            **common,
            optional_params=optional_params or {},
            debug=_is_debugging_on(),
        )
    if _declares(rust_messages, "debug"):
        return rust_messages(**common, debug=_is_debugging_on())
    return rust_messages(**common)


async def amessages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: Union[float, httpx.Timeout] | None,
    optional_params: dict[str, object] | None = None,
) -> dict[str, object] | None:
    rust_amessages = load_rust_amessages()
    if rust_amessages is None:
        return None
    common = _common_kwargs(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout=timeout,
    )
    if _declares(rust_amessages, "optional_params"):
        return await rust_amessages(
            **common,
            optional_params=optional_params or {},
            debug=_is_debugging_on(),
        )
    if _declares(rust_amessages, "debug"):
        return await rust_amessages(**common, debug=_is_debugging_on())
    return await rust_amessages(**common)
