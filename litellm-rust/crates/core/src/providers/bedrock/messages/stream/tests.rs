//! Ported from the Bedrock streaming tests in
//! `tests/test_litellm/llms/bedrock/messages/invoke_transformations/test_anthropic_claude3_transformation.py`,
//! plus cases with no Python counterpart covering behavior the ported suite
//! leaves unexercised. Ported tests name the Python test they mirror.

use aws_smithy_eventstream::frame::write_message_to;
use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
use base64::Engine;
use bytes::{Bytes, BytesMut};
use futures_util::StreamExt;
use serde_json::{Value, json};

use super::*;

/// Wire-accurate Bedrock frame: the Anthropic event base64-wrapped under
/// `bytes`, tagged with `:message-type`.
fn event_frame(message_type: &str, payload: Value) -> Vec<u8> {
    let wrapper = json!({
        "bytes": base64::engine::general_purpose::STANDARD
            .encode(serde_json::to_vec(&payload).expect("payload"))
    });
    frame_with_payload(message_type, serde_json::to_vec(&wrapper).expect("wrapper"))
}

/// Real Bedrock exceptions carry their payload raw, not base64-wrapped.
fn raw_frame(message_type: &str, payload: Value) -> Vec<u8> {
    frame_with_payload(message_type, serde_json::to_vec(&payload).expect("payload"))
}

fn frame_with_payload(message_type: &str, payload: Vec<u8>) -> Vec<u8> {
    let mut buffer = BytesMut::new();
    let message = Message::new(payload).add_header(Header::new(
        ":message-type",
        HeaderValue::String(message_type.to_string().into()),
    ));
    write_message_to(&message, &mut buffer).expect("frame");
    buffer.to_vec()
}

async fn run_pipeline(bytes: Vec<u8>, chunk_size: usize) -> String {
    let chunks: Vec<Result<Bytes, reqwest::Error>> = bytes
        .chunks(chunk_size.max(1))
        .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
        .collect();
    let collected: Vec<Bytes> = bedrock_sse_stream(futures_util::stream::iter(chunks))
        .map(|chunk| chunk.expect("chunk"))
        .collect()
        .await;
    String::from_utf8(collected.concat()).expect("utf8")
}

/// Compares semantically rather than byte-for-byte: Python pads its JSON
/// separators and preserves key insertion order, neither of which serde_json
/// reproduces.
fn parse_sse(text: &str) -> Vec<(String, Value)> {
    text.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let (event_line, data_line) = block.split_once('\n').expect("event and data lines");
            let event = event_line
                .strip_prefix("event: ")
                .expect("event line")
                .to_string();
            let data = data_line.strip_prefix("data: ").expect("data line");
            (event, serde_json::from_str(data).expect("data json"))
        })
        .collect()
}

async fn promote(events: Vec<Value>) -> Vec<Value> {
    let frames = events.into_iter().map(BedrockFrame::Event);
    promote_message_stop_usage(futures_util::stream::iter(frames))
        .filter_map(|frame| async move {
            match frame {
                BedrockFrame::Event(event) => Some(event),
                BedrockFrame::Failure(_) => None,
            }
        })
        .collect()
        .await
}

// test_chunk_parser_usage_transformation
#[test]
fn invocation_metrics_become_anthropic_usage() {
    let parsed = rename_invocation_metrics(json!({
        "type": "message_delta",
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 10, "outputTokenCount": 5}
    }));
    assert!(parsed.get("amazon-bedrock-invocationMetrics").is_none());
    assert_eq!(parsed["usage"]["input_tokens"], 10);
    assert_eq!(parsed["usage"]["output_tokens"], 5);
}

#[test]
fn events_without_invocation_metrics_are_untouched() {
    let event = json!({"type": "message_start", "message": {"id": "msg_1"}});
    assert_eq!(rename_invocation_metrics(event.clone()), event);
}

// test_promote_message_stop_usage_preserves_message_delta_output_tokens
#[tokio::test]
async fn message_stop_does_not_clobber_delta_output_tokens() {
    let promoted = promote(vec![
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"input_tokens": 3, "cache_creation_input_tokens": 10553,
                      "cache_read_input_tokens": 25490, "output_tokens": 12}
        }),
        json!({"type": "message_stop", "usage": {"input_tokens": 3, "output_tokens": 9}}),
    ])
    .await;

    let delta = &promoted[0];
    assert_eq!(delta["type"], "message_delta");
    assert_eq!(delta["usage"]["output_tokens"], 12);
    assert_eq!(delta["usage"]["cache_creation_input_tokens"], 10553);
    assert_eq!(delta["usage"]["cache_read_input_tokens"], 25490);
    assert_eq!(delta["usage"]["input_tokens"], 3);
    assert_eq!(promoted[1]["type"], "message_stop");
}

// test_promote_message_start_cache_when_message_stop_omits_cache_fields
#[tokio::test]
async fn message_start_cache_fills_gaps_left_by_message_stop() {
    // LIT-2411: some deployments report the cache breakdown only up front, and
    // missing this merge produced negative input costs downstream.
    let promoted = promote(vec![
        json!({"type": "message_start", "message": {"usage": {
            "input_tokens": 3, "cache_creation_input_tokens": 10, "cache_read_input_tokens": 25490}}}),
        json!({"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 7}}),
        json!({"type": "message_stop", "usage": {"input_tokens": 3, "output_tokens": 7}}),
    ])
    .await;

    let delta = promoted
        .iter()
        .find(|event| event["type"] == "message_delta")
        .expect("delta");
    assert_eq!(delta["usage"]["cache_read_input_tokens"], 25490);
    assert_eq!(delta["usage"]["cache_creation_input_tokens"], 10);
}

// No Python counterpart: every Python stream pairs a delta with a stop, so the
// post-loop flush never fires there.
#[tokio::test]
async fn pending_delta_is_flushed_when_the_stream_ends_without_message_stop() {
    let promoted = promote(vec![
        json!({"type": "message_start", "message": {"usage": {"cache_read_input_tokens": 99}}}),
        json!({"type": "message_delta", "usage": {"output_tokens": 4}}),
    ])
    .await;

    let delta = promoted.last().expect("delta survives");
    assert_eq!(delta["type"], "message_delta");
    assert_eq!(delta["usage"]["output_tokens"], 4);
    assert_eq!(delta["usage"]["cache_read_input_tokens"], 99);
}

// No Python counterpart: the ported tests all use equal input_tokens, so an
// implementation skipping the overwrite would still pass them.
#[tokio::test]
async fn message_stop_input_tokens_overwrite_delta_and_coerce_non_integers() {
    let promoted = promote(vec![
        json!({"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 7}}),
        json!({"type": "message_stop", "usage": {"input_tokens": 41}}),
    ])
    .await;
    assert_eq!(promoted[0]["usage"]["input_tokens"], 41);

    let coerced = promote(vec![
        json!({"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 7}}),
        json!({"type": "message_stop", "usage": {"input_tokens": "not-a-number"}}),
    ])
    .await;
    assert_eq!(coerced[0]["usage"]["input_tokens"], 0);
}

// test_bedrock_sse_wrapper_encodes_dict_chunks
#[tokio::test]
async fn events_are_encoded_as_anthropic_sse() {
    let bytes = [
        event_frame(
            "event",
            json!({"type": "message_start", "message": {"id": "msg_1"}}),
        ),
        event_frame("event", json!({"type": "message_stop"})),
    ]
    .concat();
    let sse = run_pipeline(bytes, 4096).await;

    assert!(sse.starts_with("event: message_start\n"));
    assert!(sse.ends_with("\n\n"));
    let events = parse_sse(&sse);
    assert_eq!(events[0].0, "message_start");
    assert_eq!(events[0].1["message"]["id"], "msg_1");
    assert_eq!(events[1].0, "message_stop");
}

// test_bedrock_sse_wrapper_no_error_event_when_stream_ends_with_message_stop
#[tokio::test]
async fn no_error_event_when_the_stream_ends_cleanly() {
    let bytes = [
        event_frame("event", json!({"type": "message_start", "message": {}})),
        event_frame("event", json!({"type": "message_stop"})),
    ]
    .concat();
    let events = parse_sse(&run_pipeline(bytes, 4096).await);
    assert!(events.iter().all(|(name, _)| name != "error"));
    assert_eq!(events.last().expect("last").0, "message_stop");
}

// test_bedrock_sse_wrapper_appends_error_event_when_stream_truncates_mid_tool_use
#[tokio::test]
async fn truncated_tool_use_stream_ends_with_an_error_event() {
    // LIT-3724: closing this as a clean 200 handed clients unterminated tool JSON.
    let bytes = [
        event_frame(
            "event",
            json!({"type": "message_start", "message": {"id": "msg_1"}}),
        ),
        event_frame(
            "event",
            json!({"type": "content_block_start", "index": 0,
                   "content_block": {"type": "tool_use", "id": "tooluse_1", "name": "write", "input": {}}}),
        ),
        event_frame(
            "event",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "input_json_delta", "partial_json": "{\"path\": \"/builder/docs/QUAL"}}),
        ),
    ]
    .concat();
    let events = parse_sse(&run_pipeline(bytes, 4096).await);

    assert_eq!(events.len(), 4);
    let (name, payload) = events.last().expect("error event");
    assert_eq!(name, "error");
    assert_eq!(payload["type"], "error");
    assert_eq!(payload["error"]["type"], "api_error");
}

// test_bedrock_sse_wrapper_keeps_usage_in_message_start_and_message_delta
#[tokio::test]
async fn usage_survives_end_to_end_on_message_start_and_message_delta() {
    let bytes = [
        event_frame(
            "event",
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 0}}}),
        ),
        event_frame(
            "event",
            json!({"type": "message_delta", "usage": {"input_tokens": 3, "output_tokens": 12}}),
        ),
        event_frame("event", json!({"type": "message_stop"})),
    ]
    .concat();
    let events = parse_sse(&run_pipeline(bytes, 4096).await);

    let start = events
        .iter()
        .find(|(name, _)| name == "message_start")
        .expect("start");
    assert_eq!(start.1["message"]["usage"]["input_tokens"], 3);
    let delta = events
        .iter()
        .find(|(name, _)| name == "message_delta")
        .expect("delta");
    assert_eq!(delta.1["usage"]["output_tokens"], 12);
}

// No Python counterpart: botocore owns reassembly there, this is our code here.
#[tokio::test]
async fn frames_decode_identically_at_every_split_offset() {
    let bytes = [
        event_frame(
            "event",
            json!({"type": "message_start", "message": {"id": "msg_1"}}),
        ),
        event_frame(
            "event",
            json!({"type": "content_block_delta", "index": 0,
                   "delta": {"type": "text_delta", "text": "hello"}}),
        ),
        event_frame("event", json!({"type": "message_stop"})),
    ]
    .concat();
    let expected = parse_sse(&run_pipeline(bytes.clone(), bytes.len()).await);

    for chunk_size in 1..=bytes.len() {
        let actual = parse_sse(&run_pipeline(bytes.clone(), chunk_size).await);
        assert_eq!(actual, expected, "diverged at chunk size {chunk_size}");
    }
}

#[tokio::test]
async fn exception_and_error_frames_terminate_the_stream() {
    for message_type in ["exception", "error"] {
        let bytes = [
            event_frame("event", json!({"type": "message_start", "message": {}})),
            raw_frame(message_type, json!({"message": "model unavailable"})),
        ]
        .concat();
        let events = parse_sse(&run_pipeline(bytes, 4096).await);

        let (name, payload) = events.last().expect("terminal event");
        assert_eq!(name, "error", "for :message-type {message_type}");
        assert_eq!(payload["error"]["type"], "api_error");
        assert!(
            payload["error"]["message"]
                .as_str()
                .expect("message")
                .contains("model unavailable")
        );
        // Already terminal, so no second synthetic error is appended.
        assert_eq!(events.iter().filter(|(name, _)| name == "error").count(), 1);
    }
}

#[tokio::test]
async fn empty_payload_frames_are_skipped() {
    let bytes = [
        frame_with_payload("event", Vec::new()),
        event_frame("event", json!({"type": "message_stop"})),
    ]
    .concat();
    let events = parse_sse(&run_pipeline(bytes, 4096).await);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "message_stop");
}
