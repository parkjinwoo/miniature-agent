use agent_model::{
    AnthropicBackend, Backend, ChatCompletionsBackend, OpenAiBackend,
};

use crate::provider_registry::{BackendSpec, ProviderSpec};

#[derive(Debug, Clone)]
pub(crate) enum AppBackend {
    OpenAi(OpenAiBackend),
    Anthropic(AnthropicBackend),
    ChatCompletions(ChatCompletionsBackend),
}

#[async_trait::async_trait]
impl Backend for AppBackend {
    fn name(&self) -> &'static str {
        match self {
            Self::OpenAi(backend) => backend.name(),
            Self::Anthropic(backend) => backend.name(),
            Self::ChatCompletions(backend) => backend.name(),
        }
    }

    fn supports(&self, capability: agent_model::Capability) -> bool {
        match self {
            Self::OpenAi(backend) => backend.supports(capability),
            Self::Anthropic(backend) => backend.supports(capability),
            Self::ChatCompletions(backend) => backend.supports(capability),
        }
    }

    async fn stream(
        &self,
        request: agent_model::ModelRequest,
    ) -> anyhow::Result<agent_model::ModelEventStream> {
        match self {
            Self::OpenAi(backend) => backend.stream(request).await,
            Self::Anthropic(backend) => backend.stream(request).await,
            Self::ChatCompletions(backend) => backend.stream(request).await,
        }
    }
}

pub(crate) fn configured_backend(spec: &ProviderSpec) -> AppBackend {
    match &spec.backend {
        BackendSpec::OpenAiResponses => {
            let mut backend = OpenAiBackend::new();
            if let Some(base_url) = spec.resolved_base_url() {
                backend = backend.with_base_url(base_url);
            }
            AppBackend::OpenAi(backend)
        }
        BackendSpec::AnthropicMessages => {
            let mut backend = AnthropicBackend::new();
            if let Some(base_url) = spec.resolved_base_url() {
                backend = backend.with_base_url(base_url);
            }
            AppBackend::Anthropic(backend)
        }
        BackendSpec::ChatCompletions {
            backend_name,
            default_base_url,
            compat,
        } => {
            let base_url = spec
                .resolved_base_url()
                .unwrap_or_else(|| default_base_url.clone());
            AppBackend::ChatCompletions(
                ChatCompletionsBackend::new(backend_name)
                    .with_compat(compat.clone())
                    .with_base_url(base_url),
            )
        }
    }
}
