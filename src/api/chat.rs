use futures::stream::StreamExt;
use reqwest::Client;
use reqwest_eventsource::{Event, EventSource};
use schemars::{schema::RootSchema, schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_aux::field_attributes::deserialize_default_from_empty_object;
use typed_builder::TypedBuilder;

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum ChatModel {
    #[serde(rename = "gpt-3.5-turbo-0613")]
    GPT3,
    #[serde(rename = "gpt-3.5-turbo-16k-0613")]
    GPT3_16K,
    #[serde(rename = "gpt-4")]
    GPT4_MAY,
    #[serde(rename = "gpt-4-0613")]
    GPT4,
    #[serde(rename = "gpt-4-1106-preview")]
    GPT4_TURBO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FunctionCallType {
    Auto,
    None,
    Name(String),
}

#[derive(Debug, Serialize, Deserialize, TypedBuilder)]
pub struct ChatRequest {
    #[builder(default = ChatModel::GPT4)]
    model: ChatModel,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    functions: Option<Vec<Function>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    function_call: Option<FunctionCallType>,
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

fn clamp<T: core::cmp::PartialOrd>(value: T, min: T, max: T) -> T {
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

    pub fn function_call(&self) -> Option<&FunctionCall> {
        self.message().and_then(|m| {
            if let ChatMessage::Assistant {
                content: AssistantContent::FunctionCall { function_call },
                ..
            } = m
            {
                Some(function_call)
            } else {
                None
            }
        })
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
    #[serde(deserialize_with = "deserialize_default_from_empty_object")]
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
pub struct FunctionCall {
    name: String,
    arguments: String,
}

impl FunctionCall {
    pub fn to_type<'a, T: Deserialize<'a>>(&'a self) -> Result<T, serde_json::Error> {
        serde_json::from_str(&self.arguments)
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn arguments(&self) -> String {
        self.arguments.clone()
    }
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AssistantContent {
    Content { content: String },
    FunctionCall { function_call: FunctionCall },
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
        #[serde(flatten)]
        content: AssistantContent,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// This describes the result of a function (whose name is given by the name field)
    Function {
        content: String,
        name: String,
    },
}

impl ChatMessage {
    pub fn new_user(content: impl Into<String>, name: Option<String>) -> Self {
        Self::User {
            content: content.into(),
            name,
        }
    }

    pub fn new_system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn new_assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: AssistantContent::Content {
                content: content.into(),
            },
            name: None,
        }
    }

    pub fn new_function(content: impl Into<String>, name: impl Into<String>) -> Self {
        Self::Function {
            content: content.into(),
            name: name.into(),
        }
    }
}

impl ChatMessage {
    pub fn content(&self) -> Option<String> {
        let content = match self {
            Self::User { content, .. } => content.to_string(),
            Self::System { content } => content.to_string(),
            Self::Assistant {
                content: AssistantContent::Content { content },
                ..
            } => content.to_string(),
            Self::Assistant {
                content: AssistantContent::FunctionCall { .. },
                ..
            } => return None,
            Self::Function { content, .. } => content.to_string(),
        };

        Some(content)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Function {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<RootSchema>,
}

impl Function {
    pub fn new(name: impl Into<String>, description: Option<String>) -> Self {
        Self {
            name: name.into(),
            description,
            parameters: None,
        }
    }

    pub fn from_object<T: JsonSchema>(
        name: impl Into<String>,
        description: Option<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            parameters: Some(schema_for!(T)),
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

#[derive(Debug, Clone)]
pub struct JsonResponse {
    pub antecedent: String,
    pub json: Option<String>,
}

impl JsonResponse {
    pub fn to_full_string(&self) -> String {
        match &self.json {
            Some(json) => format!("{}{}", self.antecedent, json),
            None => self.antecedent.clone(),
        }
    }

    pub fn deserialize<'de, T: Deserialize<'de>>(
        &'de self,
    ) -> Result<Option<T>, Box<dyn std::error::Error + Send + Sync>> {
        match &self.json {
            Some(json) => {
                let output: T = serde_json::from_str(json)?;
                Ok(Some(output))
            }
            None => Ok(None),
        }
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

                        if let Some(ChatDelta::Content(s)) = stream.delta() {
                            print!("{s}");
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
                    }
                }
                Err(e) => {
                    return Err(e.into());
                }
                // Err(e) => match e {
                //     reqwest_eventsource::Error::Utf8(_) => {
                //         panic!("utf8 error!")
                //     }
                //     reqwest_eventsource::Error::InvalidContentType(_) => {
                //         panic!("invalid content type!")
                //     }
                //     reqwest_eventsource::Error::InvalidLastEventId(_) => {
                //         panic!("invalid last event id!")
                //     }
                //     reqwest_eventsource::Error::InvalidStatusCode(_) => {
                //         panic!("invalid status code!")
                //     }
                //     reqwest_eventsource::Error::Parser(_) => {
                //         panic!("parser error!")
                //     }
                //     reqwest_eventsource::Error::StreamEnded => {
                //         panic!("stream ended!")
                //     }
                //     reqwest_eventsource::Error::Transport(_) => {
                //         panic!("transport error!")
                //     }
                // },
            }
        }

        Ok(JsonResponse {
            antecedent: string_response,
            json: json_response,
        })
    }
}
