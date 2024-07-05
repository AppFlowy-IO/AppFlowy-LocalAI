use anyhow::Result;
use appflowy_local_ai::llm_chat::{ChatPluginConfig, LocalChatLLMChat};
use appflowy_local_ai::llm_embedding::{EmbeddingPluginConfig, LocalEmbedding};
use appflowy_plugin::error::SidecarError;
use appflowy_plugin::manager::SidecarManager;
use bytes::Bytes;
use simsimd::SpatialSimilarity;
use std::f64;
use std::path::PathBuf;
use std::sync::{Arc, Once};
use tokio_stream::wrappers::ReceiverStream;
use tracing_subscriber::fmt::Subscriber;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct LocalAITest {
  config: LocalAIConfiguration,
  chat_manager: LocalChatLLMChat,
  embedding_manager: LocalEmbedding,
}

impl LocalAITest {
  pub fn new() -> Result<Self> {
    let config = LocalAIConfiguration::new()?;
    let sidecar = Arc::new(SidecarManager::new());
    let chat_manager = LocalChatLLMChat::new(sidecar.clone());
    let embedding_manager = LocalEmbedding::new(sidecar);
    Ok(Self {
      config,
      chat_manager,
      embedding_manager,
    })
  }

  pub async fn init_chat_plugin(&self) {
    let config = ChatPluginConfig::new(
      self.config.chat_bin_path.clone(),
      self.config.chat_model_absolute_path(),
    )
    .unwrap()
    .with_device("cpu");
    self.chat_manager.init_chat_plugin(config).await.unwrap();
  }

  pub async fn init_embedding_plugin(&self) {
    let config = EmbeddingPluginConfig::new(
      self.config.embedding_bin_path.clone(),
      self.config.embedding_model_absolute_path(),
    )
    .unwrap();
    self
      .embedding_manager
      .init_embedding_plugin(config)
      .await
      .unwrap();
  }

  pub async fn send_chat_message(&self, chat_id: &str, message: &str) -> String {
    self
      .chat_manager
      .generate_answer(chat_id, message)
      .await
      .unwrap()
  }

  pub async fn stream_chat_message(
    &self,
    chat_id: &str,
    message: &str,
  ) -> ReceiverStream<Result<Bytes, SidecarError>> {
    self
      .chat_manager
      .ask_question(chat_id, message)
      .await
      .unwrap()
  }

  pub async fn generate_embedding(&self, message: &str) -> Vec<Vec<f64>> {
    self
      .embedding_manager
      .generate_embedding(message)
      .await
      .unwrap()
  }

  pub async fn calculate_similarity(&self, message1: &str, message2: &str) -> f64 {
    let left = self
      .embedding_manager
      .generate_embedding(message1)
      .await
      .unwrap();
    let right = self
      .embedding_manager
      .generate_embedding(message2)
      .await
      .unwrap();

    let actual_embedding_flat = flatten_vec(left);
    let expected_embedding_flat = flatten_vec(right);
    let distance = f64::cosine(&actual_embedding_flat, &expected_embedding_flat)
      .expect("Vectors must be of the same length");

    distance.cos()
  }
}

// Function to flatten Vec<Vec<f64>> into Vec<f64>
fn flatten_vec(vec: Vec<Vec<f64>>) -> Vec<f64> {
  vec.into_iter().flatten().collect()
}

pub struct LocalAIConfiguration {
  model_dir: String,
  chat_bin_path: PathBuf,
  chat_model_name: String,
  embedding_bin_path: PathBuf,
  embedding_model_name: String,
}

impl LocalAIConfiguration {
  pub fn new() -> Result<Self> {
    dotenv::dotenv().ok();
    setup_log();

    // load from .env
    let model_dir = dotenv::var("LOCAL_AI_MODEL_DIR")?;
    let chat_bin_path = PathBuf::from(dotenv::var("CHAT_BIN_PATH")?);
    let chat_model_name = dotenv::var("LOCAL_AI_CHAT_MODEL_NAME")?;

    let embedding_bin_path = PathBuf::from(dotenv::var("EMBEDDING_BIN_PATH")?);
    let embedding_model_name = dotenv::var("LOCAL_AI_EMBEDDING_MODEL_NAME")?;

    Ok(Self {
      model_dir,
      chat_bin_path,
      chat_model_name,
      embedding_bin_path,
      embedding_model_name,
    })
  }

  pub fn chat_model_absolute_path(&self) -> PathBuf {
    let path = PathBuf::from(&self.model_dir);
    path.join(&self.chat_model_name)
  }

  pub fn embedding_model_absolute_path(&self) -> PathBuf {
    let path = PathBuf::from(&self.model_dir);
    path.join(&self.embedding_model_name)
  }
}

pub fn setup_log() {
  static START: Once = Once::new();
  START.call_once(|| {
    let level = "trace";
    let mut filters = vec![];
    filters.push(format!("appflowy_plugin={}", level));
    filters.push(format!("appflowy_local_ai={}", level));
    std::env::set_var("RUST_LOG", filters.join(","));

    let subscriber = Subscriber::builder()
      .with_env_filter(EnvFilter::from_default_env())
      .with_line_number(true)
      .with_ansi(true)
      .finish();
    subscriber.try_init().unwrap();
  });
}