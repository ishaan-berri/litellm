//! Ported 1:1 from
//! `tests/test_litellm/llms/bedrock/messages/invoke_transformations/test_anthropic_claude3_transformation.py`.
//! Each test names the Python test it mirrors so the two stay traceable.

use serde_json::{Value, json};

use super::body::{
    ensure_tool_names, filter_context_management, normalize_tool_input_schema_types,
    remove_custom_field_from_tools, strip_cache_control_scope_from_messages,
    strip_cache_control_scope_from_system, strip_cache_control_scope_from_tools,
};
use crate::messages::types::{AnthropicMessage, SystemPrompt};

fn tools(value: Value) -> Vec<Value> {
    value.as_array().expect("array").clone()
}

// test_bedrock_messages_strips_context_management
#[test]
fn context_management_is_dropped_when_no_edit_survives() {
    // The exact payload claude-cli 2.1.220 sends, which Bedrock 400s on with
    // "context_management: Extra inputs are not permitted".
    let claude_code = json!({"edits": [{"keep": "all", "type": "clear_thinking_20251015"}]});
    assert_eq!(filter_context_management(Some(claude_code)), None);

    // No `edits` key at all is also dropped.
    assert_eq!(filter_context_management(Some(json!({}))), None);
    assert_eq!(filter_context_management(None), None);
}

// test_bedrock_messages_filters_unsupported_context_management_edits
#[test]
fn context_management_keeps_only_supported_edits() {
    let filtered = filter_context_management(Some(json!({
        "edits": [
            {"type": "clear_thinking_20251015"},
            {"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}},
            {"type": "some_future_edit"}
        ]
    })))
    .expect("field survives");
    assert_eq!(
        filtered["edits"],
        json!([{"type": "clear_tool_uses_20250919", "keep": {"type": "tool_uses", "value": 3}}])
    );
}

// test_bedrock_messages_preserves_compact_context_management_and_adds_beta
// (the beta half is deferred; only the context_management assertions are ported)
#[test]
fn context_management_preserves_supported_edits_and_sibling_keys() {
    let filtered = filter_context_management(Some(json!({
        "edits": [{"type": "compact_20260112"}],
        "some_sibling": true
    })))
    .expect("field survives");
    assert_eq!(filtered["edits"], json!([{"type": "compact_20260112"}]));
    assert_eq!(filtered["some_sibling"], json!(true));
}

#[test]
fn context_management_leaves_non_object_values_alone() {
    // Python only rewrites a dict; anything else passes through untouched.
    assert_eq!(
        filter_context_management(Some(json!("unexpected"))),
        Some(json!("unexpected"))
    );
}

// test_remove_custom_field_from_tools
#[test]
fn custom_field_is_removed_from_every_tool() {
    let cleaned = remove_custom_field_from_tools(tools(json!([
        {"name": "Bash", "custom": {"defer_loading": true}},
        {"name": "Read"},
        "not-an-object"
    ])));
    assert_eq!(cleaned[0], json!({"name": "Bash"}));
    assert_eq!(cleaned[1], json!({"name": "Read"}));
    assert_eq!(cleaned[2], json!("not-an-object"));
}

// test_normalize_tool_input_schema_types_for_bedrock_invoke
// test_bedrock_invoke_messages_transform_converts_custom_tool_schema_type_to_object
#[test]
fn custom_schema_types_become_object_at_every_depth() {
    let normalized = normalize_tool_input_schema_types(tools(json!([{
        "name": "Bash",
        "input_schema": {
            "type": "custom",
            "properties": {"nested": {"type": "custom"}},
            "items": {"type": "custom"},
            "additionalProperties": {"type": "custom"},
            "anyOf": [{"type": "custom"}, {"type": "string"}]
        }
    }])));
    let schema = &normalized[0]["input_schema"];
    assert_eq!(schema["type"], "object");
    assert_eq!(schema["properties"]["nested"]["type"], "object");
    assert_eq!(schema["items"]["type"], "object");
    assert_eq!(schema["additionalProperties"]["type"], "object");
    assert_eq!(schema["anyOf"][0]["type"], "object");
    // Unrelated types are untouched.
    assert_eq!(schema["anyOf"][1]["type"], "string");
}

#[test]
fn schema_normalization_tolerates_missing_and_non_object_schemas() {
    let normalized =
        normalize_tool_input_schema_types(tools(json!([{"name": "a"}, {"input_schema": "nope"}])));
    assert_eq!(normalized[0], json!({"name": "a"}));
    assert_eq!(normalized[1], json!({"input_schema": "nope"}));
}

// test_ensure_bedrock_anthropic_messages_tool_names
// test_bedrock_invoke_messages_transform_adds_name_when_tool_missing_name
#[test]
fn missing_or_blank_tool_names_are_defaulted_by_index() {
    let named = ensure_tool_names(tools(json!([
        {"input_schema": {"type": "object"}},
        {"name": "   "},
        {"name": null},
        {"name": "Bash"}
    ])));
    assert_eq!(named[0]["name"], "litellm_unnamed_tool_0");
    assert_eq!(named[1]["name"], "litellm_unnamed_tool_1");
    assert_eq!(named[2]["name"], "litellm_unnamed_tool_2");
    assert_eq!(named[3]["name"], "Bash");
}

#[test]
fn tool_naming_is_idempotent() {
    // The bridge path runs this after Python already named the tool, so a second
    // pass must not renumber it.
    let once = ensure_tool_names(tools(json!([{"input_schema": {}}, {"name": "Bash"}])));
    let twice = ensure_tool_names(once.clone());
    assert_eq!(once, twice);
}

// test_remove_scope_from_cache_control
#[test]
fn cache_control_scope_is_stripped_from_system_and_messages() {
    let system: SystemPrompt = serde_json::from_value(json!([
        {"type": "text", "text": "prompt", "cache_control": {"type": "ephemeral", "scope": "global"}}
    ]))
    .expect("system");
    let stripped = strip_cache_control_scope_from_system(Some(system)).expect("system survives");
    let value = serde_json::to_value(stripped).expect("json");
    assert_eq!(value[0]["cache_control"], json!({"type": "ephemeral"}));

    let messages: Vec<AnthropicMessage> = serde_json::from_value(json!([{
        "role": "user",
        "content": [{"type": "text", "text": "hi",
                     "cache_control": {"type": "ephemeral", "scope": "global"}}]
    }]))
    .expect("messages");
    let value =
        serde_json::to_value(strip_cache_control_scope_from_messages(messages)).expect("json");
    assert_eq!(
        value[0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
}

#[test]
fn cache_control_ttl_is_preserved_pending_a_capability_source() {
    // TODO: Python drops `ttl` for pre-4.5 models via the cost map. Until `core`
    // has a capability source it is forwarded, and this pins that choice.
    let stripped = strip_cache_control_scope_from_tools(tools(json!([
        {"name": "Bash", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}
    ])));
    assert_eq!(
        stripped[0]["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );
}

#[test]
fn plain_text_system_and_content_are_left_alone() {
    let system: SystemPrompt = serde_json::from_value(json!("you are helpful")).expect("system");
    assert_eq!(
        serde_json::to_value(strip_cache_control_scope_from_system(Some(system)).expect("system"))
            .expect("json"),
        json!("you are helpful")
    );

    let messages: Vec<AnthropicMessage> =
        serde_json::from_value(json!([{"role": "user", "content": "hi"}])).expect("messages");
    assert_eq!(
        serde_json::to_value(strip_cache_control_scope_from_messages(messages)).expect("json"),
        json!([{"role": "user", "content": "hi"}])
    );
}
