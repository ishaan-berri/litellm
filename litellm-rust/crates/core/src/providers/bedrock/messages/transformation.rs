use serde_json::{Map, Value};

use crate::error::{CoreError, CoreResult};
use crate::messages::transformation::{AnthropicMessagesProviderConfig, MessagesAuthStrategy};
use crate::messages::types::AnthropicMessagesRequest;

use super::super::common_utils::{
    aws_region_from_model_arn, bedrock_invoke_model_id, is_valid_aws_region, resolve_bedrock_region,
};
use super::super::constants::{
    AWS_BEARER_TOKEN_BEDROCK, AWS_BEDROCK_RUNTIME_ENDPOINT, BEDROCK_ANTHROPIC_VERSION,
    BEDROCK_INVOKE_PATH_SUFFIX, BEDROCK_INVOKE_STREAM_PATH_SUFFIX,
    BEDROCK_RUNTIME_ENDPOINT_TEMPLATE,
};
use super::body;

pub struct BedrockMessagesConfig;

pub const BEDROCK_MESSAGES_CONFIG: BedrockMessagesConfig = BedrockMessagesConfig;

/// Top-level body keys Bedrock InvokeModel accepts for an Anthropic model.
/// Mirrors `BedrockInvokeAnthropicMessagesRequest.__annotations__`, which is the
/// allowlist Python filters against; anything else trips "Extra inputs are not
/// permitted".
const BEDROCK_INVOKE_EXTRA_ALLOWLIST: &[&str] = &["anthropic_version", "anthropic_beta"];

fn invalid_region(region: &str) -> CoreError {
    CoreError::InvalidRequest(format!(
        "Invalid AWS region format: {region:?}. Region names must contain only \
         lowercase letters, digits, and hyphens."
    ))
}

/// Mirrors `_get_aws_region_name`: the explicit param is validated on the way in
/// and the resolved value again on the way out, so a bad env var still fails.
fn resolve_region(
    model: &str,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    if let Some(region) = optional_params
        .get("aws_region_name")
        .and_then(Value::as_str)
        && !is_valid_aws_region(region)
    {
        return Err(invalid_region(region));
    }
    let region = resolve_bedrock_region(
        aws_region_from_model_arn(model).as_deref(),
        optional_params,
        env_lookup,
    );
    if !is_valid_aws_region(&region) {
        return Err(invalid_region(&region));
    }
    Ok(region)
}

/// Mirrors `BaseAWSLLM.get_runtime_endpoint`.
///
/// TODO: Python tests `api_base is not None`, so an empty string is used as-is
/// and yields a hostless URL. Kept for parity; the Anthropic config in this
/// workspace instead treats blank as absent.
fn runtime_endpoint(
    api_base: Option<&str>,
    optional_params: &Map<String, Value>,
    region: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    if let Some(api_base) = api_base {
        return api_base.to_string();
    }
    optional_params
        .get("aws_bedrock_runtime_endpoint")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", region))
}

pub fn complete_bedrock_url(
    api_base: Option<&str>,
    model: &str,
    optional_params: &Map<String, Value>,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let region = resolve_region(model, optional_params, env_lookup)?;
    let endpoint = runtime_endpoint(api_base, optional_params, &region, env_lookup);
    let model_id = bedrock_invoke_model_id(model, optional_params);
    // TODO: Python exempts `ai21` from the streaming suffix. Only Claude models
    // reach this config, so that branch is unreachable here.
    let suffix = if stream {
        BEDROCK_INVOKE_STREAM_PATH_SUFFIX
    } else {
        BEDROCK_INVOKE_PATH_SUFFIX
    };
    Ok(format!("{endpoint}/model/{model_id}/{suffix}"))
}

impl AnthropicMessagesProviderConfig for BedrockMessagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_bedrock_url(api_base, model, optional_params, stream, env_lookup)
    }

    /// Mirrors the bearer branch of `BaseAWSLLM._sign_request`.
    ///
    /// TODO: Python falls back to SigV4 when no bearer token is present. This
    /// path has no signer, so it errors instead; the Python caller gates the
    /// bridge to bearer requests so the fallback is never silently wrong.
    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .or_else(|| {
                env_lookup(AWS_BEARER_TOKEN_BEDROCK).filter(|value| !value.trim().is_empty())
            })
            .ok_or_else(|| {
                CoreError::Auth(
                    "Missing Bedrock bearer token - set `api_key` or the \
                     AWS_BEARER_TOKEN_BEDROCK environment variable"
                        .to_string(),
                )
            })
    }

    fn auth_strategy(&self) -> MessagesAuthStrategy {
        MessagesAuthStrategy::Bearer
    }

    fn accepts_bearer_auth(&self) -> bool {
        true
    }

    /// Python's Bedrock `validate_anthropic_messages_environment` is a no-op and
    /// its signer adds only Content-Type and Authorization, so the shared
    /// `anthropic-version` default must not leak onto this path.
    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[]
    }

    fn transform_request(
        &self,
        request: AnthropicMessagesRequest,
    ) -> CoreResult<AnthropicMessagesRequest> {
        let mut extra = request.extra;
        extra.retain(|key, _| BEDROCK_INVOKE_EXTRA_ALLOWLIST.contains(&key.as_str()));
        extra
            .entry("anthropic_version")
            .or_insert_with(|| Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()));

        // Nested cleanups run before the top-level allowlist below, which keeps
        // `tools` wholesale and so cannot reach inside it.
        let tools = request.tools.map(|tools| {
            body::strip_cache_control_scope_from_tools(body::ensure_tool_names(
                body::normalize_tool_input_schema_types(body::remove_custom_field_from_tools(
                    tools,
                )),
            ))
        });

        Ok(AnthropicMessagesRequest {
            // Bedrock takes the model in the URL path and rejects it in the body.
            model: String::new(),
            // The verb, not a body field; the caller picks the path suffix.
            stream: None,
            // Anthropic-only extensions Bedrock InvokeModel does not accept.
            service_tier: None,
            container: None,
            mcp_servers: None,
            output_format: None,
            speed: None,
            inference_geo: None,
            tools,
            context_management: body::filter_context_management(request.context_management),
            system: body::strip_cache_control_scope_from_system(request.system),
            messages: body::strip_cache_control_scope_from_messages(request.messages),
            extra,
            ..request
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::types::AnthropicMessagesResponse;
    use crate::providers::bedrock::constants::{AWS_REGION, AWS_REGION_NAME};
    use serde_json::json;

    const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

    fn request(value: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(value).expect("valid request")
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn params(pairs: &[(&str, Value)]) -> Map<String, Value> {
        pairs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect()
    }

    #[test]
    fn builds_default_and_streaming_urls_with_encoded_arn() {
        let env = |key: &str| (key == AWS_REGION).then(|| "eu-west-1".to_string());
        let model = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/foo/bar";
        // The ARN carries its own region, which outranks the environment in Python.
        assert_eq!(
            complete_bedrock_url(None, model, &Map::new(), false, &env).expect("url"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Ffoo%2Fbar/invoke"
        );
        assert!(
            complete_bedrock_url(None, "claude", &Map::new(), true, &env)
                .expect("url")
                .ends_with("/invoke-with-response-stream")
        );
    }

    #[test]
    fn url_honors_explicit_region_and_runtime_endpoint_params() {
        let env = |key: &str| (key == AWS_REGION_NAME).then(|| "us-west-2".to_string());
        assert_eq!(
            complete_bedrock_url(
                None,
                "claude",
                &params(&[("aws_region_name", json!("ap-south-1"))]),
                false,
                &env
            )
            .expect("url"),
            "https://bedrock-runtime.ap-south-1.amazonaws.com/model/claude/invoke"
        );
        assert_eq!(
            complete_bedrock_url(
                None,
                "claude",
                &params(&[(
                    "aws_bedrock_runtime_endpoint",
                    json!("http://127.0.0.1:8080")
                )]),
                false,
                &env
            )
            .expect("url"),
            "http://127.0.0.1:8080/model/claude/invoke"
        );
        // api_base outranks every other endpoint source.
        assert_eq!(
            complete_bedrock_url(
                Some("http://localhost:1"),
                "claude",
                &params(&[(
                    "aws_bedrock_runtime_endpoint",
                    json!("http://127.0.0.1:8080")
                )]),
                false,
                &env
            )
            .expect("url"),
            "http://localhost:1/model/claude/invoke"
        );
    }

    #[test]
    fn trait_complete_url_delegates_without_streaming() {
        let env = |key: &str| (key == AWS_REGION_NAME).then(|| "us-west-2".to_string());
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .complete_url(None, "claude", &Map::new(), false, &env)
                .expect("url"),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/claude/invoke"
        );
    }

    #[test]
    fn malformed_regions_are_rejected() {
        assert!(
            complete_bedrock_url(
                None,
                "claude",
                &params(&[("aws_region_name", json!("US-WEST-2"))]),
                false,
                &no_env
            )
            .is_err()
        );
        assert!(
            complete_bedrock_url(None, "claude", &Map::new(), false, &|key| (key
                == AWS_REGION_NAME)
                .then(|| "not a region".to_string()))
            .is_err()
        );
    }

    #[test]
    fn empty_model_mirrors_pythons_permissive_url() {
        // TODO: Python builds this malformed URL rather than failing fast. Kept
        // for parity; erroring here would be the better behavior.
        assert_eq!(
            complete_bedrock_url(None, " ", &Map::new(), false, &|key| (key
                == AWS_REGION_NAME)
                .then(|| "us-west-2".to_string()))
            .expect("url"),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/ /invoke"
        );
    }

    #[test]
    fn bearer_token_resolution_matches_python() {
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(Some("from-arg"), &no_env)
                .expect("key"),
            "from-arg"
        );
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(None, &|key| (key == AWS_BEARER_TOKEN_BEDROCK)
                    .then(|| "from-env".to_string()))
                .expect("key"),
            "from-env"
        );
        assert!(
            BEDROCK_MESSAGES_CONFIG
                .resolve_api_key(None, &no_env)
                .is_err()
        );
    }

    #[test]
    fn auth_is_bearer_and_no_anthropic_version_header_is_sent() {
        assert_eq!(
            BEDROCK_MESSAGES_CONFIG.auth_strategy(),
            MessagesAuthStrategy::Bearer
        );
        assert!(BEDROCK_MESSAGES_CONFIG.default_headers().is_empty());
    }

    #[test]
    fn request_removes_path_and_unsupported_fields() {
        let transformed = BEDROCK_MESSAGES_CONFIG
            .transform_request(request(json!({
                "model": "claude",
                "stream": true,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {"user_id": "kept"},
                "speed": "fast",
                "mcp_servers": [{"name": "x"}],
                "output_format": {"type": "json_schema"},
                "unknown_future_field": true,
                "anthropic_beta": ["compact-2026-01-12"],
                "tools": [{"name": "search"}]
            })))
            .expect("transform");
        let value = serde_json::to_value(transformed).expect("json");

        assert!(value.get("model").is_none());
        assert!(value.get("stream").is_none());
        assert!(value.get("speed").is_none());
        assert!(value.get("mcp_servers").is_none());
        assert!(value.get("output_format").is_none());
        assert!(value.get("unknown_future_field").is_none());
        assert_eq!(value["anthropic_version"], ANTHROPIC_VERSION);
        assert_eq!(value["anthropic_beta"], json!(["compact-2026-01-12"]));
        assert!(value.get("tools").is_some());
        // Python's allowlist keeps `metadata`; dropping it would diverge.
        assert_eq!(value["metadata"], json!({"user_id": "kept"}));
    }

    #[test]
    fn caller_supplied_anthropic_version_is_preserved() {
        let transformed = BEDROCK_MESSAGES_CONFIG
            .transform_request(request(json!({
                "anthropic_version": "bedrock-2099-01-01",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .expect("transform");
        assert_eq!(
            serde_json::to_value(transformed).expect("json")["anthropic_version"],
            "bedrock-2099-01-01"
        );
    }

    /// The payload claude-cli 2.1.220 actually sends, reduced to the fields that
    /// exercise a transform. Bedrock 400s on the `context_management` edit.
    fn claude_code_request() -> Value {
        json!({
            "model": "claude",
            "stream": true,
            "max_tokens": 10,
            "context_management": {"edits": [{"keep": "all", "type": "clear_thinking_20251015"}]},
            "system": [{"type": "text", "text": "sys",
                        "cache_control": {"type": "ephemeral", "scope": "global"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi",
                 "cache_control": {"type": "ephemeral", "scope": "global"}}]}],
            "tools": [{"input_schema": {"type": "custom"}, "custom": {"defer_loading": true}}],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "metadata": {"user_id": "kept"}
        })
    }

    #[test]
    fn claude_code_payload_is_shaped_for_bedrock() {
        let value = serde_json::to_value(
            BEDROCK_MESSAGES_CONFIG
                .transform_request(request(claude_code_request()))
                .expect("transform"),
        )
        .expect("json");

        // The 400: the only edit Claude Code sends is unsupported, so the whole
        // field must go rather than be forwarded empty.
        assert!(value.get("context_management").is_none());
        assert!(value["tools"][0].get("custom").is_none());
        assert_eq!(value["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(value["tools"][0]["name"], "litellm_unnamed_tool_0");
        assert_eq!(
            value["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(
            value["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(value["metadata"], json!({"user_id": "kept"}));
        assert_eq!(value["anthropic_version"], ANTHROPIC_VERSION);
    }

    #[test]
    fn transform_request_is_idempotent() {
        // The Python bridge hands Rust an already-transformed body, so a second
        // pass must be a no-op or the wire request diverges.
        let once = BEDROCK_MESSAGES_CONFIG
            .transform_request(request(claude_code_request()))
            .expect("transform");
        let twice = BEDROCK_MESSAGES_CONFIG
            .transform_request(once.clone())
            .expect("transform");
        assert_eq!(
            serde_json::to_value(once).expect("json"),
            serde_json::to_value(twice).expect("json")
        );
    }

    #[test]
    fn response_passes_through_untouched() {
        // Python returns the provider payload as-is; restamping `model` here
        // would diverge from that passthrough.
        let payload = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5-20250929",
            "content": [],
            "stop_reason": null,
            "stop_sequence": null
        });
        let response: AnthropicMessagesResponse =
            serde_json::from_value(payload.clone()).expect("response");
        let transformed = BEDROCK_MESSAGES_CONFIG
            .transform_response("us.anthropic.claude-sonnet-4-5-20250929-v1:0", response)
            .expect("response");
        assert_eq!(serde_json::to_value(transformed).expect("json"), payload);
    }
}
