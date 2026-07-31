//! Bedrock `invoke-with-response-stream` answers with
//! `application/vnd.amazon.eventstream`, a binary framing format, not SSE. This
//! module is the Rust mirror of Python's
//! `AmazonAnthropicClaudeMessagesConfig.get_async_streaming_response_iterator`
//! and converts that framing into the Anthropic SSE protocol.
//!
//! The pipeline order mirrors Python and is load-bearing: usage renaming feeds
//! usage promotion, which buffers a `message_delta` until it has seen the next
//! event, which in turn feeds the encoder.
//!
//! Two deliberate divergences from Python, both forced:
//!
//! 1. `serde_json` renders compact and sorts object keys, where Python's
//!    `json.dumps` pads separators and preserves insertion order. The documents
//!    are equivalent; matching bytes would mean `preserve_order` workspace-wide.
//! 2. A frame carrying an exception terminates the stream with an Anthropic
//!    `event: error`. Python raises and the proxy emits a bare `data:` line, but
//!    Claude Code speaks the Anthropic protocol, which has no such shape.

use base64::Engine;
use bytes::{Bytes, BytesMut};
use futures_util::stream::{self, BoxStream};
use futures_util::{Stream, StreamExt};
use serde_json::{Map, Value};

use aws_smithy_eventstream::frame::{DecodedFrame, MessageFrameDecoder};

const MESSAGE_TYPE_HEADER: &str = ":message-type";
const EVENT_MESSAGE_TYPE: &str = "event";
const TERMINAL_EVENT_TYPE: &str = "message_stop";
const ERROR_EVENT_TYPE: &str = "error";

/// Mirrors `INCOMPLETE_STREAM_ERROR_MESSAGE` in `streaming_iterator.py`.
const INCOMPLETE_STREAM_ERROR: &str = "Provider stream ended before emitting a message_stop event; \
     the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated.";

/// One decoded upstream frame, before any Anthropic-level normalization.
enum BedrockFrame {
    /// A normal `event` frame carrying one Anthropic event.
    Event(Value),
    /// An exception or error frame. Terminates the stream.
    Failure(String),
}

/// The full pipeline: raw upstream bytes in, Anthropic SSE bytes out.
pub fn bedrock_sse_stream<S>(upstream: S) -> BoxStream<'static, Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    encode_sse(promote_message_stop_usage(decode_frames(upstream).map(
        |frame| match frame {
            BedrockFrame::Event(event) => BedrockFrame::Event(rename_invocation_metrics(event)),
            failure => failure,
        },
    )))
}

/// Stage a. Mirrors `AWSEventStreamDecoder.aiter_bytes`: reassemble frames from
/// arbitrarily chunked bytes, then unwrap each payload.
fn decode_frames<S>(upstream: S) -> impl Stream<Item = BedrockFrame> + Send + 'static
where
    S: Stream<Item = Result<Bytes, reqwest::Error>> + Send + 'static,
{
    struct State<S> {
        upstream: S,
        buffer: BytesMut,
        decoder: MessageFrameDecoder,
        pending: Vec<BedrockFrame>,
        finished: bool,
    }

    stream::unfold(
        State {
            upstream: Box::pin(upstream),
            buffer: BytesMut::new(),
            decoder: MessageFrameDecoder::new(),
            pending: Vec::new(),
            finished: false,
        },
        |mut state| async move {
            loop {
                if !state.pending.is_empty() {
                    return Some((state.pending.remove(0), state));
                }
                if state.finished {
                    return None;
                }
                match state.upstream.next().await {
                    Some(Ok(chunk)) => {
                        state.buffer.extend_from_slice(&chunk);
                        state.pending = drain_frames(&mut state.decoder, &mut state.buffer);
                    }
                    Some(Err(error)) => {
                        // A mid-stream transport failure still ends with a
                        // protocol-level event rather than a truncated body.
                        state.finished = true;
                        state.pending = vec![BedrockFrame::Failure(error.to_string())];
                    }
                    None => state.finished = true,
                }
            }
        },
    )
}

/// Pull every complete frame currently sitting in the buffer. A partial frame
/// stays buffered until more bytes arrive, which is why the upstream may split
/// a frame at any byte offset.
fn drain_frames(decoder: &mut MessageFrameDecoder, buffer: &mut BytesMut) -> Vec<BedrockFrame> {
    let mut frames = Vec::new();
    loop {
        match decoder.decode_frame(&mut *buffer) {
            Ok(DecodedFrame::Complete(message)) => {
                let message_type = message
                    .headers()
                    .iter()
                    .find(|header| header.name().as_str() == MESSAGE_TYPE_HEADER)
                    .and_then(|header| header.value().as_string().ok())
                    .map(|value| value.as_str().to_string());
                let payload = message.payload().as_ref();
                match message_type.as_deref() {
                    Some(EVENT_MESSAGE_TYPE) | None => match unwrap_event(payload) {
                        Some(event) => frames.push(BedrockFrame::Event(event)),
                        // Mirrors `_parse_message_from_event` skipping empty chunks.
                        None => continue,
                    },
                    Some(_) => frames.push(BedrockFrame::Failure(
                        String::from_utf8_lossy(payload).to_string(),
                    )),
                }
            }
            Ok(DecodedFrame::Incomplete) => return frames,
            Err(error) => {
                frames.push(BedrockFrame::Failure(error.to_string()));
                return frames;
            }
        }
    }
}

/// Bedrock wraps each Anthropic event as `{"bytes": "<base64>"}`; some frames
/// carry the payload directly, which `_parse_message_from_event` also handles.
fn unwrap_event(payload: &[u8]) -> Option<Value> {
    if payload.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_slice(payload).ok()?;
    let Some(encoded) = value.get("bytes").and_then(Value::as_str) else {
        return Some(value);
    };
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .ok()?;
    if decoded.is_empty() {
        return None;
    }
    serde_json::from_slice(&decoded).ok()
}

/// Stage b. Mirrors `AmazonAnthropicClaudeMessagesStreamDecoder._chunk_parser`:
/// Bedrock reports usage in camelCase under its own key.
fn rename_invocation_metrics(mut event: Value) -> Value {
    let Some(object) = event.as_object_mut() else {
        return event;
    };
    let Some(metrics) = object.remove("amazon-bedrock-invocationMetrics") else {
        return event;
    };
    let usage: Map<String, Value> = [
        ("input_tokens", "inputTokenCount"),
        ("output_tokens", "outputTokenCount"),
    ]
    .into_iter()
    .filter_map(|(anthropic, bedrock)| {
        metrics
            .get(bedrock)
            .cloned()
            .map(|value| (anthropic.to_string(), value))
    })
    .collect();
    if !usage.is_empty() {
        object.insert("usage".to_string(), Value::Object(usage));
    }
    event
}

/// Stage c. Mirrors `_promote_message_stop_usage`: hold back `message_delta`
/// so the usage the client sees is self-consistent with `message_stop` and,
/// where the provider only reports cache tokens up front, `message_start`.
fn promote_message_stop_usage<S>(events: S) -> impl Stream<Item = BedrockFrame> + Send + 'static
where
    S: Stream<Item = BedrockFrame> + Send + 'static,
{
    struct State<S> {
        events: S,
        pending_delta: Option<Value>,
        start_usage: Option<Value>,
        queue: Vec<BedrockFrame>,
        finished: bool,
    }

    stream::unfold(
        State {
            events: Box::pin(events),
            pending_delta: None,
            start_usage: None,
            queue: Vec::new(),
            finished: false,
        },
        |mut state| async move {
            loop {
                if !state.queue.is_empty() {
                    return Some((state.queue.remove(0), state));
                }
                if state.finished {
                    return None;
                }
                match state.events.next().await {
                    Some(BedrockFrame::Event(event)) => {
                        let event_type = event
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        match event_type.as_str() {
                            "message_start" => {
                                state.start_usage = event
                                    .get("message")
                                    .and_then(|message| message.get("usage"))
                                    .cloned();
                                state.queue = flush_pending(&mut state.pending_delta, None);
                                state.queue.push(BedrockFrame::Event(event));
                            }
                            "message_delta" => state.pending_delta = Some(event),
                            "message_stop" if state.pending_delta.is_some() => {
                                let mut delta =
                                    state.pending_delta.take().expect("checked by the guard");
                                merge_stop_usage(&mut delta, &event, state.start_usage.as_ref());
                                state.queue =
                                    vec![BedrockFrame::Event(delta), BedrockFrame::Event(event)];
                            }
                            _ => {
                                state.queue = flush_pending(&mut state.pending_delta, None);
                                state.queue.push(BedrockFrame::Event(event));
                            }
                        }
                    }
                    Some(failure) => {
                        state.queue = flush_pending(&mut state.pending_delta, None);
                        state.queue.push(failure);
                    }
                    None => {
                        state.finished = true;
                        // Python's post-loop flush: a delta held when the stream
                        // ends still has to reach the client, cache-merged.
                        state.queue =
                            flush_pending(&mut state.pending_delta, state.start_usage.as_ref());
                    }
                }
            }
        },
    )
}

fn flush_pending(pending: &mut Option<Value>, start_usage: Option<&Value>) -> Vec<BedrockFrame> {
    let Some(mut delta) = pending.take() else {
        return Vec::new();
    };
    if let Some(start_usage) = start_usage {
        let mut usage = delta
            .get("usage")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        merge_start_cache(&mut usage, start_usage);
        if !usage.is_empty() {
            set_usage(&mut delta, usage);
        }
    }
    vec![BedrockFrame::Event(delta)]
}

/// Fields `message_stop` may restate, plus the cache breakdown it usually owns.
const CACHE_FIELDS: &[&str] = &["cache_creation_input_tokens", "cache_read_input_tokens"];

fn merge_stop_usage(delta: &mut Value, stop: &Value, start_usage: Option<&Value>) {
    let stop_usage = stop.get("usage").and_then(Value::as_object).cloned();
    let mut usage = delta
        .get("usage")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if let Some(stop_usage) = stop_usage.as_ref() {
        for field in CACHE_FIELDS {
            if let Some(value) = stop_usage.get(*field) {
                usage.insert((*field).to_string(), value.clone());
            }
        }
        // Python takes message_stop's input_tokens verbatim, coercing a
        // non-integer to 0 rather than leaving the delta's value in place.
        if let Some(raw_input) = stop_usage.get("input_tokens") {
            let coerced = if raw_input.is_i64() || raw_input.is_u64() {
                raw_input.clone()
            } else {
                Value::from(0)
            };
            usage.insert("input_tokens".to_string(), coerced);
        }
    }
    if let Some(start_usage) = start_usage {
        merge_start_cache(&mut usage, start_usage);
    }
    if !usage.is_empty() {
        set_usage(delta, usage);
    }
}

/// Only fills fields the delta is missing; some deployments report the cache
/// breakdown solely on `message_start`.
fn merge_start_cache(usage: &mut Map<String, Value>, start_usage: &Value) {
    let Some(start) = start_usage.as_object() else {
        return;
    };
    for field in CACHE_FIELDS
        .iter()
        .chain(std::iter::once(&"cache_creation"))
    {
        if !usage.contains_key(*field)
            && let Some(value) = start.get(*field)
        {
            usage.insert((*field).to_string(), value.clone());
        }
    }
}

fn set_usage(event: &mut Value, usage: Map<String, Value>) {
    if let Some(object) = event.as_object_mut() {
        object.insert("usage".to_string(), Value::Object(usage));
    }
}

/// Stage d and e. Mirrors `BaseAnthropicMessagesStreamingIterator.async_sse_wrapper`.
/// Everything reaching here is already a decoded value, so unlike Python there is
/// no raw-bytes passthrough branch to mirror.
fn encode_sse<S>(events: S) -> BoxStream<'static, Result<Bytes, std::io::Error>>
where
    S: Stream<Item = BedrockFrame> + Send + 'static,
{
    struct State<S> {
        events: S,
        saw_terminal: bool,
        finished: bool,
    }

    stream::unfold(
        State {
            events: Box::pin(events),
            saw_terminal: false,
            finished: false,
        },
        |mut state| async move {
            if state.finished {
                return None;
            }
            match state.events.next().await {
                Some(BedrockFrame::Event(event)) => {
                    let event_type = event
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("message");
                    state.saw_terminal = state.saw_terminal
                        || event_type == TERMINAL_EVENT_TYPE
                        || event_type == ERROR_EVENT_TYPE;
                    Some((Ok(sse_frame(event_type, &event)), state))
                }
                Some(BedrockFrame::Failure(message)) => {
                    state.saw_terminal = true;
                    Some((Ok(error_frame(&message)), state))
                }
                None => {
                    state.finished = true;
                    // LIT-3724: a stream that dies mid tool_use must not look
                    // like a clean end, or clients keep truncated tool JSON.
                    (!state.saw_terminal).then(|| (Ok(error_frame(INCOMPLETE_STREAM_ERROR)), state))
                }
            }
        },
    )
    .boxed()
}

fn sse_frame(event_type: &str, payload: &Value) -> Bytes {
    Bytes::from(format!("event: {event_type}\ndata: {payload}\n\n"))
}

fn error_frame(message: &str) -> Bytes {
    let payload = serde_json::json!({
        "type": "error",
        "error": {"type": "api_error", "message": message},
    });
    sse_frame(ERROR_EVENT_TYPE, &payload)
}

#[cfg(test)]
mod tests;
