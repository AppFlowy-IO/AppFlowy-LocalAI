use anyhow::Result;
use appflowy_local_ai::llm_chat::{ChatPluginConfig, LocalChatLLMChat};
use appflowy_local_ai::llm_embedding::{EmbeddingPluginConfig, LocalEmbedding};
use appflowy_plugin::error::PluginError;
use appflowy_plugin::manager::PluginManager;
use bytes::Bytes;
use simsimd::SpatialSimilarity;
use std::f64;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use tokio_stream::wrappers::ReceiverStream;
use tracing_subscriber::fmt::Subscriber;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

pub struct LocalAITest {
  config: LocalAIConfiguration,
  pub chat_manager: LocalChatLLMChat,
  pub embedding_manager: LocalEmbedding,
}

impl LocalAITest {
  pub fn new() -> Result<Self> {
    let config = LocalAIConfiguration::new()?;
    let sidecar = Arc::new(PluginManager::new());
    let chat_manager = LocalChatLLMChat::new(sidecar.clone());
    let embedding_manager = LocalEmbedding::new(sidecar);
    Ok(Self {
      config,
      chat_manager,
      embedding_manager,
    })
  }

  pub async fn init_chat_plugin(&self) {
    let mut config = ChatPluginConfig::new(
      self.config.chat_bin_path.clone(),
      self.config.chat_model_absolute_path(),
    )
    .unwrap()
    .with_device("cpu");

    if let Some(related_question_model) = self.config.related_question_model_absolute_path() {
      config = config.with_related_model_path(related_question_model);
    }

    let persist_dir = tempfile::tempdir().unwrap().path().to_path_buf();
    config
      .set_rag_enabled(&self.config.embedding_model_absolute_path(), &persist_dir)
      .unwrap();

    self.chat_manager.init_chat_plugin(config).await.unwrap();
  }

  pub async fn init_embedding_plugin(&self) {
    let temp_dir = tempfile::tempdir().unwrap().path().to_path_buf();
    let config = EmbeddingPluginConfig::new(
      self.config.embedding_bin_path.clone(),
      self.config.embedding_model_absolute_path(),
      Some(temp_dir),
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
      .ask_question(chat_id, message)
      .await
      .unwrap()
  }

  pub async fn stream_chat_message(
    &self,
    chat_id: &str,
    message: &str,
  ) -> ReceiverStream<Result<Bytes, PluginError>> {
    self
      .chat_manager
      .stream_question(chat_id, message)
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

  pub async fn calculate_similarity(&self, input: &str, expected: &str) -> f64 {
    let left = self
      .embedding_manager
      .generate_embedding(input)
      .await
      .unwrap();
    let right = self
      .embedding_manager
      .generate_embedding(expected)
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
  chat_model: String,
  related_question_model: Option<String>,
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
    let chat_model = dotenv::var("LOCAL_AI_CHAT_MODEL_NAME")?;
    let related_question_model = dotenv::var("LOCAL_AI_RELATED_QUESTION_NAME").ok();
    let embedding_bin_path = PathBuf::from(dotenv::var("EMBEDDING_BIN_PATH")?);
    let embedding_model_name = dotenv::var("LOCAL_AI_EMBEDDING_MODEL_NAME")?;

    Ok(Self {
      model_dir,
      chat_bin_path,
      chat_model,
      related_question_model,
      embedding_bin_path,
      embedding_model_name,
    })
  }

  pub fn chat_model_absolute_path(&self) -> PathBuf {
    let path = PathBuf::from(&self.model_dir);
    path.join(&self.chat_model)
  }

  pub fn related_question_model_absolute_path(&self) -> Option<PathBuf> {
    self.related_question_model.as_ref().map(|model| {
      let path = PathBuf::from(&self.model_dir);
      path.join(model)
    })
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

pub fn get_asset_path(name: &str) -> PathBuf {
  let file = format!("tests/asset/{name}");
  let absolute_path = std::env::current_dir().unwrap().join(Path::new(&file));
  absolute_path
}
