use anyhow::anyhow;
use appflowy_plugin::core::parser::{DefaultResponseParser, ResponseParser};
use appflowy_plugin::core::plugin::Plugin;
use appflowy_plugin::error::{PluginError, RemoteError};
use bytes::Bytes;
use serde_json::json;
use serde_json::Value as JsonValue;
use std::sync::Weak;
use tokio_stream::wrappers::ReceiverStream;
use tracing::instrument;

pub struct ChatPluginOperation {
  plugin: Weak<Plugin>,
}

impl ChatPluginOperation {
  pub fn new(plugin: Weak<Plugin>) -> Self {
    ChatPluginOperation { plugin }
  }

  fn get_plugin(&self) -> Result<std::sync::Arc<Plugin>, PluginError> {
    self
      .plugin
      .upgrade()
      .ok_or_else(|| PluginError::Internal(anyhow!("Plugin is dropped")))
  }

  async fn send_request<T: ResponseParser>(
    &self,
    method: &str,
    params: JsonValue,
  ) -> Result<T::ValueType, PluginError> {
    let plugin = self.get_plugin()?;
    let mut request = json!({ "method": method });
    request
      .as_object_mut()
      .unwrap()
      .extend(params.as_object().unwrap().clone());
    plugin.async_request::<T>("handle", &request).await
  }

  pub async fn create_chat(&self, chat_id: &str, rag_enabled: bool) -> Result<(), PluginError> {
    self
      .send_request::<DefaultResponseParser>(
        "create_chat",
        json!({ "chat_id": chat_id, "rag_enabled": rag_enabled }),
      )
      .await
  }

  pub async fn close_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    self
      .send_request::<DefaultResponseParser>("close_chat", json!({ "chat_id": chat_id }))
      .await
  }

  pub async fn send_message(
    &self,
    chat_id: &str,
    message: &str,
    rag_enabled: bool,
  ) -> Result<String, PluginError> {
    self
      .send_request::<ChatResponseParser>(
        "answer",
        json!({ "chat_id": chat_id, "params": { "content": message, "rag_enabled": rag_enabled } }),
      )
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn stream_message(
    &self,
    chat_id: &str,
    message: &str,
    rag_enabled: bool,
  ) -> Result<ReceiverStream<Result<Bytes, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;
    let params = json!({
        "chat_id": chat_id,
        "method": "stream_answer",
        "params": { "content": message, "rag_enabled": rag_enabled }
    });
    plugin.stream_request::<ChatStreamResponseParser>("handle", &params)
  }

  pub async fn get_related_questions(&self, chat_id: &str) -> Result<Vec<String>, PluginError> {
    self
      .send_request::<ChatRelatedQuestionsResponseParser>(
        "related_question",
        json!({ "chat_id": chat_id }),
      )
      .await
  }

  pub async fn index_file(&self, chat_id: &str, file_path: &str) -> Result<(), PluginError> {
    let params = json!({ "file_path": file_path, "metadatas": [{"chat_id": chat_id}] });
    self
      .send_request::<DefaultResponseParser>(
        "index_file",
        json!({ "chat_id": chat_id, "params": params }),
      )
      .await
  }

  #[instrument(level = "debug", skip(self), err)]
  pub async fn complete_text(
    &self,
    message: &str,
    complete_type: CompleteTextType,
  ) -> Result<ReceiverStream<Result<Bytes, PluginError>>, PluginError> {
    let plugin = self.get_plugin()?;
    let complete_type = complete_type as u8;
    let params = json!({
        "method": "complete_text",
        "params": { "text": message, "type": complete_type }
    });
    plugin.stream_request::<ChatStreamResponseParser>("handle", &params)
  }
}

pub struct ChatResponseParser;
impl ResponseParser for ChatResponseParser {
  type ValueType = String;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| data.as_str())
      .map(String::from)
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct ChatStreamResponseParser;
impl ResponseParser for ChatStreamResponseParser {
  type ValueType = Bytes;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .as_str()
      .map(|message| Bytes::from(message.to_string()))
      .ok_or(RemoteError::ParseResponse(json))
  }
}

pub struct ChatRelatedQuestionsResponseParser;
impl ResponseParser for ChatRelatedQuestionsResponseParser {
  type ValueType = Vec<String>;

  fn parse_json(json: JsonValue) -> Result<Self::ValueType, RemoteError> {
    json
      .get("data")
      .and_then(|data| data.as_array())
      .map(|array| {
        array
          .iter()
          .filter_map(|item| item.get("content").map(|s| s.to_string()))
          .collect()
      })
      .ok_or(RemoteError::ParseResponse(json))
  }
}

#[derive(Debug, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum CompleteTextType {
  ImproveWriting = 1,
  SpellingAndGrammar = 2,
  MakeShorter = 3,
  MakeLonger = 4,
  AskAI = 5,
}

impl From<i8> for CompleteTextType {
  fn from(value: i8) -> Self {
    match value {
      1 => CompleteTextType::ImproveWriting,
      2 => CompleteTextType::SpellingAndGrammar,
      3 => CompleteTextType::MakeShorter,
      4 => CompleteTextType::MakeLonger,
      5 => CompleteTextType::AskAI,
      _ => CompleteTextType::ImproveWriting,
    }
  }
}
