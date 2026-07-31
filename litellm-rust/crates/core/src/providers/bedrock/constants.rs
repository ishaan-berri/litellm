pub const AWS_ACCESS_KEY_ID: &str = "AWS_ACCESS_KEY_ID";
pub const AWS_SECRET_ACCESS_KEY: &str = "AWS_SECRET_ACCESS_KEY";
pub const AWS_SESSION_TOKEN: &str = "AWS_SESSION_TOKEN";
pub const AWS_REGION_NAME: &str = "AWS_REGION_NAME";
pub const AWS_REGION: &str = "AWS_REGION";
pub const AWS_SESSION_NAME: &str = "AWS_SESSION_NAME";
pub const AWS_PROFILE_NAME: &str = "AWS_PROFILE_NAME";
pub const AWS_ROLE_NAME: &str = "AWS_ROLE_NAME";
pub const AWS_WEB_IDENTITY_TOKEN: &str = "AWS_WEB_IDENTITY_TOKEN";
pub const AWS_ROLE_ARN: &str = "AWS_ROLE_ARN";
pub const AWS_WEB_IDENTITY_TOKEN_FILE: &str = "AWS_WEB_IDENTITY_TOKEN_FILE";
pub const AWS_STS_ENDPOINT: &str = "AWS_STS_ENDPOINT";
pub const AWS_EXTERNAL_ID: &str = "AWS_EXTERNAL_ID";
pub const AWS_BEARER_TOKEN_BEDROCK: &str = "AWS_BEARER_TOKEN_BEDROCK";
pub const AWS_BEDROCK_RUNTIME_ENDPOINT: &str = "AWS_BEDROCK_RUNTIME_ENDPOINT";
pub const BEDROCK_SERVICE: &str = "bedrock";
pub const DEFAULT_SESSION_NAME_PREFIX: &str = "litellm-session";
pub const DEFAULT_BEDROCK_REGION: &str = "us-west-2";
pub const BEDROCK_RUNTIME_ENDPOINT_TEMPLATE: &str =
    "https://bedrock-runtime.{region}.amazonaws.com";
pub const BEDROCK_INVOKE_PATH_SUFFIX: &str = "invoke";
pub const BEDROCK_INVOKE_STREAM_PATH_SUFFIX: &str = "invoke-with-response-stream";
pub const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Stripped in order with no early exit, mirroring Python's
/// `strip_bedrock_routing_prefix`, so compound prefixes fully unwind.
pub const BEDROCK_ROUTING_PREFIXES: &[&str] = &[
    "bedrock/",
    "converse/",
    "invoke/",
    "openai/",
    "nova-2/",
    "nova/",
];
