use axum::{
    body::Full,
    body::HttpBody,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use typed_builder::TypedBuilder;
use url::Url;
use utoipa::openapi::OpenApi;

pub mod api;

pub use api::chat::{ChatMessage, ChatRequest};
pub use api::embeddings::{knn_search, string_embeddings, EmbeddingRequest};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ManifestAuth {
    None,
    UserHttp,
    ServiceHttp,
    Oauth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ManifestApi {
    Openapi {
        url: String,
        is_user_authenticated: bool,
    },
}

const MAX_NAME_FOR_HUMAN: usize = 20;
const MAX_NAME_FOR_MODEL: usize = 50;
const MAX_DESCRIPTION_FOR_HUMAN: usize = 100;
const MAX_DESCRIPTION_FOR_MODEL: usize = 8000;

fn test_len(string: impl Into<String>, iden: &str, len: usize) -> String {
    let string: String = string.into();
    if string.len() > len {
        panic!(
            "{} too long (expected <= {}, got {})",
            iden,
            len,
            string.len()
        );
    }
    string
}

#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
pub struct Manifest {
    #[builder(setter(into))]
    pub schema_version: String,

    #[builder(setter(transform = |n: impl Into<String>| test_len(n, "name_for_human", MAX_NAME_FOR_HUMAN)))]
    pub name_for_human: String,

    #[builder(setter(transform = |n: impl Into<String>| test_len(n, "name_for_model", MAX_NAME_FOR_MODEL)))]
    pub name_for_model: String,

    #[builder(setter(transform = |d: impl Into<String>| test_len(d, "description_for_human", MAX_DESCRIPTION_FOR_HUMAN)))]
    pub description_for_human: String,

    #[builder(setter(transform = |d: impl Into<String>| test_len(d, "description_for_model", MAX_DESCRIPTION_FOR_MODEL)))]
    pub description_for_model: String,
    pub auth: ManifestAuth,
    pub api: ManifestApi,

    #[builder(setter(into))]
    pub logo_url: String,

    #[builder(setter(into))]
    pub contact_email: String,

    #[builder(setter(into))]
    pub legal_info_url: String,
}

struct ServeState {
    manifest: Manifest,
    openapi: OpenApi,
    logo: Vec<u8>,
}

pub fn serve_plugin_info<B>(manifest: Manifest, api: OpenApi, icon_path: &str) -> Router<(), B>
where
    B: HttpBody + Send + 'static,
{
    let ManifestApi::Openapi { url, .. } = &manifest.api;
    let url = Url::parse(url).expect("error parsing API URL");
    let api_route = url.path();

    let url = Url::parse(&manifest.logo_url).expect("error parsing icon URL");
    let icon_route = url.path();

    let state = Arc::new(ServeState {
        manifest,
        openapi: api,
        logo: std::fs::read(icon_path).expect("error reading logo file"),
    });

    Router::new()
        .route("/.well-known/ai-plugin.json", get(serve_manifest))
        .route(api_route, get(serve_api_docs))
        .route(icon_route, get(serve_icon))
        .with_state(state)
}

async fn serve_manifest(State(state): State<Arc<ServeState>>) -> Json<Manifest> {
    Json::from(state.manifest.clone())
}

async fn serve_api_docs(
    State(state): State<Arc<ServeState>>,
) -> Result<impl IntoResponse, StatusCode> {
    Ok(Response::builder()
        .header("Content-Type", "application/yaml")
        .body(Full::from(
            state
                .openapi
                .to_yaml()
                .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
        ))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
}

async fn serve_icon(State(state): State<Arc<ServeState>>) -> Result<impl IntoResponse, StatusCode> {
    Ok(Response::builder()
        .header("Content-Type", "image/png")
        .body(Full::from(state.logo.clone()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_max_len() {
        let port = 3030;
        let base_url = format!("http://localhost:{port}");

        Manifest::builder()
            .schema_version(env!("CARGO_PKG_VERSION"))
            .name_for_human("To-Do Plugin Name that is Way TOO LONG!!!")
            .name_for_model("todo")
            .description_for_human(
                "Plugin for managing a TODO list, you can add, remove and view your TODOs.",
            )
            .description_for_model(
                "Plugin for managing a TODO list, you can add, remove and view your TODOs.",
            )
            .auth(ManifestAuth::None)
            .api(ManifestApi::Openapi {
                url: format!("{base_url}/openapi.yaml"),
                is_user_authenticated: false,
            })
            .logo_url(format!("{base_url}/logo.png"))
            .contact_email("support@example.com")
            .legal_info_url("http://example.com/legal")
            .build();
    }
}
