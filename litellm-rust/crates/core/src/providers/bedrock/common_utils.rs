//! Bedrock helpers shared by more than one route.
//!
//! Mirrors the pieces of Python's `BaseAWSLLM` that are route-independent:
//! region resolution and the routing-prefix / ARN handling that turns a LiteLLM
//! model string into a Bedrock model id.

use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde_json::{Map, Value};

use super::constants::{
    AWS_REGION, AWS_REGION_NAME, BEDROCK_ROUTING_PREFIXES, DEFAULT_BEDROCK_REGION,
};

/// `urllib.parse.quote(value, safe="")` leaves RFC 3986 unreserved characters
/// alone; `NON_ALPHANUMERIC` would additionally encode these four.
const PYTHON_QUOTE_SET: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'.')
    .remove(b'_')
    .remove(b'~');

const ARN_PREFIX: &str = "arn:";
const BEDROCK_ARN_MARKER: &str = "arn:aws:bedrock";

/// Mirrors Python's `BaseAWSLLM.encode_model_id`.
pub fn percent_encode_model_id(model_id: &str) -> String {
    utf8_percent_encode(model_id, PYTHON_QUOTE_SET).to_string()
}

/// Mirrors `strip_bedrock_routing_prefix`: every prefix is tested in order
/// against the current value, so `bedrock/invoke/arn:...` unwinds fully.
pub fn strip_bedrock_routing_prefix(model: &str) -> &str {
    BEDROCK_ROUTING_PREFIXES
        .iter()
        .fold(model, |model, prefix| {
            model.strip_prefix(prefix).unwrap_or(model)
        })
}

/// Mirrors `BaseAWSLLM._validate_aws_region_name`'s `\A[a-z0-9-]+\Z`.
pub fn is_valid_aws_region(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

/// Mirrors `BaseAWSLLM._get_aws_region_from_model_arn`: index 3 of the *whole*
/// ARN, rejected unless it looks like a region.
///
/// TODO: `bedrock_model_id_and_region` in `audio_transcription.rs` strips the
/// `arn:` prefix before indexing and so returns the account id here. It is
/// untested and left alone for now; fixing it changes transcription routing.
pub fn aws_region_from_model_arn(model: &str) -> Option<String> {
    if !model.contains(BEDROCK_ARN_MARKER) {
        return None;
    }
    model
        .split(':')
        .nth(3)
        .filter(|region| is_valid_aws_region(region))
        .map(str::to_string)
}

/// Precedence from `BaseAWSLLM._get_aws_region_name`: explicit param, then the
/// model ARN, then either region env var, then Python's hardcoded default.
///
/// TODO: Python has one more step before the default,
/// `boto3.Session().region_name`, which reads `~/.aws/config` and `AWS_PROFILE`.
/// `core` may not touch the filesystem (see `crates/core/CLAUDE.md`), so a host
/// with no region env but a configured AWS profile resolves differently here.
pub fn resolve_bedrock_region(
    model_region: Option<&str>,
    optional_params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    if let Some(region) = optional_params
        .get("aws_region_name")
        .and_then(Value::as_str)
    {
        return region.to_string();
    }
    if let Some(region) = model_region {
        return region.to_string();
    }
    env_lookup(AWS_REGION_NAME)
        .or_else(|| env_lookup(AWS_REGION))
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string())
}

/// Mirrors `BaseAWSLLM.get_bedrock_model_id` for the Anthropic invoke path.
///
/// TODO: Python additionally rewrites the id for `llama` / `deepseek_r1` /
/// `openai` invoke providers. Only Claude models reach the messages config, so
/// those branches are unreachable here.
pub fn bedrock_invoke_model_id(model: &str, optional_params: &Map<String, Value>) -> String {
    if let Some(model_id) = optional_params.get("model_id").and_then(Value::as_str) {
        return percent_encode_model_id(model_id).replacen("invoke/", "", 1);
    }
    let stripped = strip_bedrock_routing_prefix(model);
    if stripped.starts_with(ARN_PREFIX) {
        return percent_encode_model_id(stripped);
    }
    stripped.replacen("invoke/", "", 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const ARN: &str = "arn:aws:bedrock:us-east-1:086734376398:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0";

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn strips_compound_routing_prefixes() {
        assert_eq!(
            strip_bedrock_routing_prefix("bedrock/invoke/anthropic.claude-v2"),
            "anthropic.claude-v2"
        );
        assert_eq!(
            strip_bedrock_routing_prefix("invoke/anthropic.claude-v2"),
            "anthropic.claude-v2"
        );
        assert_eq!(
            strip_bedrock_routing_prefix("anthropic.claude-v2"),
            "anthropic.claude-v2"
        );
    }

    #[test]
    fn encodes_arns_but_leaves_plain_model_ids_literal() {
        let params = Map::new();
        let encoded = bedrock_invoke_model_id(&format!("bedrock/{ARN}"), &params);
        assert!(!encoded.contains("bedrock/"));
        assert!(encoded.contains("%3A"), "colons must be encoded: {encoded}");
        assert!(
            encoded.contains("%2F"),
            "slashes must be encoded: {encoded}"
        );

        assert_eq!(
            bedrock_invoke_model_id(&format!("bedrock/invoke/{ARN}"), &params),
            encoded,
            "compound prefix must strip to the same id"
        );

        // Python leaves a non-ARN id untouched, colons included.
        assert_eq!(
            bedrock_invoke_model_id("us.anthropic.claude-sonnet-4-5-20250929-v1:0", &params),
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        );
    }

    #[test]
    fn explicit_model_id_param_wins_and_is_encoded() {
        let params = Map::from_iter([("model_id".to_string(), json!(ARN))]);
        let model_id = bedrock_invoke_model_id("ignored.model", &params);
        assert!(model_id.contains("%3A"));
        assert!(!model_id.contains("ignored.model"));
    }

    #[test]
    fn arn_region_comes_from_index_three_of_the_whole_arn() {
        assert_eq!(aws_region_from_model_arn(ARN).as_deref(), Some("us-east-1"));
        assert_eq!(aws_region_from_model_arn("anthropic.claude-v2"), None);
        // Not a bedrock ARN, so Python returns None rather than guessing.
        assert_eq!(aws_region_from_model_arn("arn:aws:s3:us-east-1:1:x"), None);
    }

    #[test]
    fn region_precedence_matches_python() {
        let params = Map::from_iter([("aws_region_name".to_string(), json!("eu-west-1"))]);
        assert_eq!(
            resolve_bedrock_region(Some("us-east-1"), &params, &no_env),
            "eu-west-1"
        );

        let empty = Map::new();
        // The ARN's region outranks both env vars, matching Python's ordering.
        assert_eq!(
            resolve_bedrock_region(Some("us-east-1"), &empty, &|key| (key == AWS_REGION)
                .then(|| "eu-west-1".to_string())),
            "us-east-1"
        );
        assert_eq!(
            resolve_bedrock_region(None, &empty, &|key| (key == AWS_REGION_NAME)
                .then(|| "ap-south-1".to_string())),
            "ap-south-1"
        );
        assert_eq!(
            resolve_bedrock_region(None, &empty, &|key| (key == AWS_REGION)
                .then(|| "eu-west-1".to_string())),
            "eu-west-1"
        );
        assert_eq!(
            resolve_bedrock_region(None, &empty, &no_env),
            DEFAULT_BEDROCK_REGION
        );
    }

    #[test]
    fn region_validation_matches_pythons_pattern() {
        assert!(is_valid_aws_region("us-west-2"));
        assert!(!is_valid_aws_region("US-WEST-2"));
        assert!(!is_valid_aws_region("us_west_2"));
        assert!(!is_valid_aws_region(""));
    }
}
