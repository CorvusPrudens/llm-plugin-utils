use reqwest::Client;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use typed_builder::TypedBuilder;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum ChatModel {
    #[serde(rename = "gpt-3.5-turbo")]
    GPT3,
    #[serde(rename = "gpt-4")]
    GPT4,
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct ChatRequest {
    #[builder(default = ChatModel::GPT4)]
    model: ChatModel,
    messages: Vec<ChatMessage>,
    #[builder(default = 0.7, setter(transform = |f: f32| clamp(f, 0., 2.)))]
    temperature: f32,
    #[builder(default = false)]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    stop: Option<Vec<String>>,
    #[builder(default = 0., setter(transform = |f: f32| clamp(f, -2., 2.)))]
    frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    max_tokens: Option<usize>,
}

pub fn clamp<T: core::cmp::PartialOrd>(value: T, min: T, max: T) -> T {
    if value > max {
        return max;
    }
    if value < min {
        return min;
    }
    value
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatResponse {
    id: String,
    object: String,
    created: u64,
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Debug, Deserialize)]
pub struct ChatStream {
    id: String,
    object: String,
    created: u64,
    choices: Vec<StreamChoice>,
}

impl ChatResponse {
    pub fn message(&self) -> Option<&ChatMessage> {
        self.choices.get(0).map(|c| &c.message)
    }

    pub fn messages(&self) -> Vec<&ChatMessage> {
        self.choices.iter().map(|c| &c.message).collect()
    }

    pub fn tokens(&self) -> ChatUsage {
        self.usage
    }
}

impl ChatStream {
    pub fn delta(&self) -> Option<ChatDelta> {
        self.choices.get(0).map(|c| c.delta.clone())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatChoice {
    index: u32,
    message: ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StreamChoice {
    index: u32,
    delta: ChatDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize, Serialize, Clone, Copy)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    User {
        content: String,
        name: Option<String>,
    },
    System {
        content: String,
    },
    Assistant {
        content: String,
    },
}

impl ChatMessage {
    pub fn content(&self) -> &str {
        match self {
            Self::User { content, name: _ } => content,
            Self::System { content } => content,
            Self::Assistant { content } => content,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChatDelta {
    Role(String),
    Content(String),
    None,
}

impl<'de> Deserialize<'de> for ChatDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let msg = Value::deserialize(deserializer)?;

        match msg.get("role") {
            Some(Value::String(s)) => return Ok(Self::Role(s.clone())),
            Some(_) => return Err(serde::de::Error::custom("expected role string")),
            None => {}
        }

        match msg.get("content") {
            Some(Value::String(s)) => return Ok(Self::Content(s.clone())),
            Some(_) => return Err(serde::de::Error::custom("expected content string")),
            None => {}
        }

        Ok(ChatDelta::None)
    }
}

impl ChatRequest {
    pub async fn request(
        self,
        client: &Client,
        api_key: &str,
    ) -> Result<ChatResponse, Box<dyn std::error::Error + Send + Sync>> {
        let response = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&self)
            .send()
            .await?
            .error_for_status()?;

        Ok(response.json::<ChatResponse>().await?)
    }
}
