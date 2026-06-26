use std::sync::Arc;

use litellm_ai_gateway::integrations::custom_guardrail::{
    CustomGuardrail, GuardrailContext, GuardrailDecision, GuardrailError, GuardrailEventHook,
    GuardrailFuture, GuardrailRequest,
};
use litellm_ai_gateway::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, LogError, LogFuture, ModelCallDetails,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use serde_json::{json, Value};

fn dropped_log_error(message: impl Into<String>) -> LogError {
    LogError {
        message: message.into(),
        kind: "PythonCallbackError".to_string(),
    }
}

fn py_to_json(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Value> {
    let json = py.import("json")?;
    let encoded: String = json.call_method1("dumps", (value,))?.extract()?;
    serde_json::from_str(&encoded).map_err(|err| PyValueError::new_err(err.to_string()))
}

fn json_to_py(py: Python<'_>, value: Value) -> PyResult<Py<PyAny>> {
    let json = py.import("json")?;
    let encoded =
        serde_json::to_string(&value).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(json.call_method1("loads", (encoded,))?.unbind())
}

fn py_attr_string(py: Python<'_>, obj: &Py<PyAny>, attr_name: &str) -> Option<String> {
    let attr = obj.bind(py).getattr(attr_name).ok()?;
    if attr.is_none() {
        return None;
    }
    if let Ok(value_attr) = attr.getattr("value") {
        if let Ok(value) = value_attr.extract::<String>() {
            return Some(value);
        }
    }
    attr.extract::<String>()
        .ok()
        .or_else(|| attr.str().ok().map(|value| value.to_string()))
}

fn py_guardrail_name(py: Python<'_>, obj: &Py<PyAny>) -> String {
    py_attr_string(py, obj, "guardrail_name").unwrap_or_else(|| {
        obj.bind(py)
            .get_type()
            .name()
            .map(|name| name.to_string())
            .unwrap_or_else(|_| "python_guardrail".to_string())
    })
}

fn guardrail_event_hook_from_str(value: &str) -> Option<GuardrailEventHook> {
    match value {
        "pre_call" | "GuardrailEventHooks.pre_call" => Some(GuardrailEventHook::PreCall),
        "during_call" | "GuardrailEventHooks.during_call" => Some(GuardrailEventHook::DuringCall),
        _ => None,
    }
}

fn py_guardrail_hooks(py: Python<'_>, obj: &Py<PyAny>) -> Vec<GuardrailEventHook> {
    let Some(attr) = obj.bind(py).getattr("event_hook").ok() else {
        return vec![GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall];
    };
    if attr.is_none() {
        return vec![GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall];
    }
    if let Ok(list) = attr.downcast::<PyList>() {
        return list
            .iter()
            .filter_map(|item| {
                let value = item
                    .getattr("value")
                    .ok()
                    .and_then(|value_attr| value_attr.extract::<String>().ok())
                    .or_else(|| item.extract::<String>().ok())
                    .or_else(|| item.str().ok().map(|value| value.to_string()))?;
                guardrail_event_hook_from_str(&value)
            })
            .collect();
    }
    let value = attr
        .getattr("value")
        .ok()
        .and_then(|value_attr| value_attr.extract::<String>().ok())
        .or_else(|| attr.extract::<String>().ok())
        .or_else(|| attr.str().ok().map(|value| value.to_string()));
    value
        .as_deref()
        .and_then(guardrail_event_hook_from_str)
        .map(|hook| vec![hook])
        .unwrap_or_else(|| vec![GuardrailEventHook::PreCall, GuardrailEventHook::DuringCall])
}

fn model_call_details_json(details: &ModelCallDetails) -> Value {
    json!({
        "model": details.model,
        "custom_llm_provider": details.custom_llm_provider,
        "call_type": details.call_type.to_string(),
        "litellm_call_id": details.litellm_call_id,
        "request_id": details.request_id,
        "metadata": {
            "user_api_key_hash": details.metadata.user_api_key_hash,
            "user_api_key_user_id": details.metadata.user_api_key_user_id,
            "user_api_key_team_id": details.metadata.user_api_key_team_id,
        },
        "standard_logging_object": details.standard_logging_payload,
    })
}

async fn call_python_awaitable(
    obj: Py<PyAny>,
    method_name: &'static str,
    args: Vec<Py<PyAny>>,
    kwargs: Option<Py<PyDict>>,
) -> PyResult<Py<PyAny>> {
    let awaitable = Python::with_gil(|py| {
        let callable = obj.bind(py).getattr(method_name)?;
        let args_tuple = pyo3::types::PyTuple::new(py, args.iter().map(|arg| arg.bind(py)))?;
        callable
            .call(args_tuple, kwargs.as_ref().map(|dict| dict.bind(py)))
            .map(|value| value.unbind())
    })?;
    let future =
        Python::with_gil(|py| pyo3_async_runtimes::tokio::into_future(awaitable.into_bound(py)))?;
    future.await
}

pub struct PythonCustomLoggerAdapter {
    obj: Py<PyAny>,
}

impl PythonCustomLoggerAdapter {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

impl CustomLogger for PythonCustomLoggerAdapter {
    fn async_log_success_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: &'a CallbackValue,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let details = model_call_details_json(model_call_details);
        let response = json!({
            "object": response_obj.object,
            "value": response_obj.value,
        });
        Box::pin(async move {
            let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
                Ok(vec![
                    json_to_py(py, details)?,
                    json_to_py(py, response)?,
                    timing.start_time.into_pyobject(py)?.unbind().into_any(),
                    timing.end_time.into_pyobject(py)?.unbind().into_any(),
                ])
            })
            .map_err(|err| dropped_log_error(err.to_string()))?;
            call_python_awaitable(obj, "async_log_success_event", args, None)
                .await
                .map_err(|err| dropped_log_error(err.to_string()))?;
            Ok(())
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        model_call_details: &'a ModelCallDetails,
        response_obj: Option<&'a CallbackValue>,
        timing: CallbackTiming,
    ) -> LogFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let details = model_call_details_json(model_call_details);
        let response = response_obj.map(|value| {
            json!({
                "object": value.object,
                "value": value.value,
            })
        });
        Box::pin(async move {
            let args = Python::with_gil(|py| -> PyResult<Vec<Py<PyAny>>> {
                Ok(vec![
                    json_to_py(py, details)?,
                    match response {
                        Some(response) => json_to_py(py, response)?,
                        None => py.None(),
                    },
                    timing.start_time.into_pyobject(py)?.unbind().into_any(),
                    timing.end_time.into_pyobject(py)?.unbind().into_any(),
                ])
            })
            .map_err(|err| dropped_log_error(err.to_string()))?;
            call_python_awaitable(obj, "async_log_failure_event", args, None)
                .await
                .map_err(|err| dropped_log_error(err.to_string()))?;
            Ok(())
        })
    }
}

pub struct PythonCustomGuardrailAdapter {
    obj: Py<PyAny>,
    name: String,
    hooks: Vec<GuardrailEventHook>,
}

impl PythonCustomGuardrailAdapter {
    pub fn new(py: Python<'_>, obj: Py<PyAny>) -> Self {
        let name = py_guardrail_name(py, &obj);
        let hooks = py_guardrail_hooks(py, &obj);
        Self { obj, name, hooks }
    }

    fn call_guardrail_hook<'a>(
        &'a self,
        method_name: &'static str,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        let obj = Python::with_gil(|py| self.obj.clone_ref(py));
        let call_type = context.call_type.to_string();
        Box::pin(async move {
            let request_for_fallback = request.clone();
            let (kwargs, data) = Python::with_gil(|py| -> PyResult<(Py<PyDict>, Py<PyAny>)> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("user_api_key_dict", py.None())?;
                kwargs.set_item("cache", py.None())?;
                let data = json_to_py(py, request.data)?;
                kwargs.set_item("data", data.bind(py))?;
                kwargs.set_item("call_type", call_type)?;
                Ok((kwargs.unbind(), data))
            })
            .map_err(|err| GuardrailError::blocked(err.to_string()))?;
            let result = call_python_awaitable(obj, method_name, Vec::new(), Some(kwargs))
                .await
                .map_err(|err| GuardrailError::blocked(err.to_string()))?;
            Python::with_gil(|py| -> Result<GuardrailDecision, GuardrailError> {
                let result = result.bind(py);
                if result.is_none() {
                    return Ok(GuardrailDecision::Allow(request_for_fallback));
                }
                if let Ok(message) = result.extract::<String>() {
                    return Err(GuardrailError::blocked(message));
                }
                let value = py_to_json(py, result)
                    .map_err(|err| GuardrailError::blocked(err.to_string()))?;
                if value.is_object() {
                    Ok(GuardrailDecision::Mask(GuardrailRequest::new(value)))
                } else {
                    let fallback = py_to_json(py, data.bind(py))
                        .map_err(|err| GuardrailError::blocked(err.to_string()))?;
                    Ok(GuardrailDecision::Mask(GuardrailRequest::new(fallback)))
                }
            })
        })
    }
}

impl CustomGuardrail for PythonCustomGuardrailAdapter {
    fn guardrail_name(&self) -> &str {
        &self.name
    }

    fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
        &self.hooks
    }

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.call_guardrail_hook("async_pre_call_hook", context, request)
    }

    fn async_moderation_hook<'a>(
        &'a self,
        context: &'a GuardrailContext,
        request: GuardrailRequest,
    ) -> GuardrailFuture<'a> {
        self.call_guardrail_hook("async_moderation_hook", context, request)
    }
}

pub fn py_callbacks_to_rust(
    py: Python<'_>,
    callbacks: Option<Py<PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomLogger>>> {
    let Some(callbacks) = callbacks else {
        return Ok(Vec::new());
    };
    callbacks
        .bind(py)
        .try_iter()?
        .map(|callback| {
            let callback = callback?;
            Ok(Arc::new(PythonCustomLoggerAdapter::new(callback.unbind()))
                as Arc<dyn CustomLogger>)
        })
        .collect()
}

pub fn py_guardrails_to_rust(
    py: Python<'_>,
    guardrails: Option<Py<PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomGuardrail>>> {
    let Some(guardrails) = guardrails else {
        return Ok(Vec::new());
    };
    guardrails
        .bind(py)
        .try_iter()?
        .map(|guardrail| {
            let guardrail = guardrail?;
            Ok(
                Arc::new(PythonCustomGuardrailAdapter::new(py, guardrail.unbind()))
                    as Arc<dyn CustomGuardrail>,
            )
        })
        .collect()
}
