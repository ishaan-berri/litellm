//! Body cleanups Bedrock InvokeModel requires beyond the top-level allowlist.
//!
//! Each function mirrors one step of Python's
//! `AmazonAnthropicClaudeMessagesConfig.transform_anthropic_messages_request`.
//! They reach inside `tools`, `messages` and `system`, which the top-level
//! allowlist cannot.
//!
//! All of them must stay idempotent: the Python SDK bridge already applies its
//! own version before handing the body over, so on that path these run second.

use serde_json::{Map, Value};

use crate::messages::types::{AnthropicMessage, ContentBlock, MessageContent, SystemPrompt};

/// `context_management.edits` types Bedrock InvokeModel accepts, mirroring
/// `_BEDROCK_INVOKE_SUPPORTED_CONTEXT_MANAGEMENT_EDITS`. Claude Code sends
/// `clear_thinking_20251015`, which is LiteLLM-internal and 400s here.
const SUPPORTED_CONTEXT_MANAGEMENT_EDITS: &[&str] =
    &["compact_20260112", "clear_tool_uses_20250919"];

const UNNAMED_TOOL_PREFIX: &str = "litellm_unnamed_tool_";

/// Mirrors `_filter_context_management_for_bedrock_invoke`: keep the supported
/// edits, drop the field entirely when none survive.
///
/// TODO: Python also adds an `anthropic-beta` value per surviving edit
/// (`compact-2026-01-12`, `context-management-2025-06-27`). That needs the beta
/// machinery, which is deferred with the rest of the model-capability work.
pub fn filter_context_management(context_management: Option<Value>) -> Option<Value> {
    let value = context_management?;
    // Python only rewrites a dict; anything else is left untouched.
    let Some(object) = value.as_object() else {
        return Some(value);
    };
    // No `edits` array means nothing Bedrock accepts, so the field goes.
    let edits = object.get("edits").and_then(Value::as_array)?;
    let retained: Vec<Value> = edits
        .iter()
        .filter(|edit| {
            edit.get("type")
                .and_then(Value::as_str)
                .is_some_and(|edit_type| SUPPORTED_CONTEXT_MANAGEMENT_EDITS.contains(&edit_type))
        })
        .cloned()
        .collect();
    if retained.is_empty() {
        return None;
    }
    let mut filtered = object.clone();
    filtered.insert("edits".to_string(), Value::Array(retained));
    Some(Value::Object(filtered))
}

/// Mirrors `remove_custom_field_from_tools`: Claude Code sends
/// `custom: {defer_loading: true}`, which Anthropic accepts and Bedrock rejects.
pub fn remove_custom_field_from_tools(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .map(|mut tool| {
            if let Some(object) = tool.as_object_mut() {
                object.remove("custom");
            }
            tool
        })
        .collect()
}

/// Mirrors `normalize_tool_input_schema_types_for_bedrock_invoke`.
pub fn normalize_tool_input_schema_types(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .map(|mut tool| {
            if let Some(schema) = tool.get_mut("input_schema") {
                normalize_custom_types_to_object(schema);
            }
            tool
        })
        .collect()
}

/// Iterative rather than recursive so a deeply nested schema cannot overflow the
/// stack on provider-controlled input.
fn normalize_custom_types_to_object(schema: &mut Value) {
    let mut stack: Vec<&mut Value> = vec![schema];
    while let Some(node) = stack.pop() {
        let Some(object) = node.as_object_mut() else {
            continue;
        };
        if object.get("type").and_then(Value::as_str) == Some("custom") {
            object.insert("type".to_string(), Value::String("object".to_string()));
        }
        for (key, child) in object {
            match key.as_str() {
                "items" | "additionalProperties" if child.is_object() => stack.push(child),
                "properties" => {
                    if let Some(properties) = child.as_object_mut() {
                        stack.extend(properties.values_mut().filter(|value| value.is_object()));
                    }
                }
                "allOf" | "anyOf" | "oneOf" => {
                    if let Some(variants) = child.as_array_mut() {
                        stack.extend(variants.iter_mut().filter(|value| value.is_object()));
                    }
                }
                _ => {}
            }
        }
    }
}

/// Mirrors `ensure_bedrock_anthropic_messages_tool_names`: Bedrock requires a
/// `name` on every tool. Re-running this must not rename an already-named tool.
pub fn ensure_tool_names(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .enumerate()
        .map(|(index, mut tool)| {
            if let Some(object) = tool.as_object_mut()
                && tool_name_is_blank(object)
            {
                object.insert(
                    "name".to_string(),
                    Value::String(format!("{UNNAMED_TOOL_PREFIX}{index}")),
                );
            }
            tool
        })
        .collect()
}

fn tool_name_is_blank(tool: &Map<String, Value>) -> bool {
    match tool.get("name") {
        None | Some(Value::Null) => true,
        Some(Value::String(name)) => name.trim().is_empty(),
        Some(_) => false,
    }
}

/// Mirrors the `scope` half of `_remove_ttl_from_cache_control`. Bedrock has no
/// cross-request cache scope and rejects the field.
///
/// TODO: Python also drops `ttl` unless the model is Claude 4.5+, which it reads
/// from `model_prices_and_context_window.json`. `core` has no capability source,
/// so `ttl` is forwarded for now.
pub fn strip_cache_control_scope_from_tools(tools: Vec<Value>) -> Vec<Value> {
    tools
        .into_iter()
        .map(|mut tool| {
            if let Some(cache_control) =
                tool.get_mut("cache_control").and_then(Value::as_object_mut)
            {
                cache_control.remove("scope");
            }
            tool
        })
        .collect()
}

pub fn strip_cache_control_scope_from_system(system: Option<SystemPrompt>) -> Option<SystemPrompt> {
    match system? {
        SystemPrompt::Blocks(blocks) => Some(SystemPrompt::Blocks(strip_scope_from_blocks(blocks))),
        text @ SystemPrompt::Text(_) => Some(text),
    }
}

pub fn strip_cache_control_scope_from_messages(
    messages: Vec<AnthropicMessage>,
) -> Vec<AnthropicMessage> {
    messages
        .into_iter()
        .map(|message| match message.content {
            MessageContent::Blocks(blocks) => AnthropicMessage {
                content: MessageContent::Blocks(strip_scope_from_blocks(blocks)),
                ..message
            },
            MessageContent::Text(_) => message,
        })
        .collect()
}

fn strip_scope_from_blocks(blocks: Vec<ContentBlock>) -> Vec<ContentBlock> {
    blocks
        .into_iter()
        .map(|mut block| {
            if let Some(cache_control) = block.cache_control.as_mut() {
                cache_control.scope = None;
            }
            block
        })
        .collect()
}
