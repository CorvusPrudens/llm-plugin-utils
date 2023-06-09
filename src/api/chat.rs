use futures::stream::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
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
        self.choices.get(0).and_then(|c| c.delta.clone())
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
    delta: Option<ChatDelta>,
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
        #[serde(skip_serializing_if = "Option::is_none")]
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

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatDelta {
    Role(String),
    Content(String),
    // None,
}

#[derive(Debug)]
pub struct JsonResponse {
    pub antecedent: String,
    pub json: Option<String>,
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

    pub async fn stream_json(
        self,
        client: &Client,
        api_key: &str,
    ) -> Result<JsonResponse, Box<dyn std::error::Error + Send + Sync>> {
        if !self.stream {
            return Err("\"stream\" must be set to true".into());
        }

        let client = client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&self);

        let mut state = super::parsing::JsonState::Idle;
        let mut es = EventSource::new(client)?;

        let mut string_response = String::new();
        let mut json_response = None;

        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => {}
                Ok(Event::Message(message)) => {
                    if message.data == "[DONE]" {
                        es.close();
                        break;
                    } else {
                        let stream: crate::api::chat::ChatStream =
                            serde_json::from_str(&message.data)?;
                        let delta = stream.delta().ok_or("error getting delta")?;

                        match delta {
                            ChatDelta::Content(s) => {
                                let (new_state, json, filtered) =
                                    super::parsing::parse_json_from_stream(&s, state);
                                state = new_state;
                                string_response.push_str(&filtered);

                                if let Some(json) = json {
                                    json_response = Some(json);
                                    es.close();
                                    break;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }

        Ok(JsonResponse {
            antecedent: string_response,
            json: json_response,
        })
    }
}
