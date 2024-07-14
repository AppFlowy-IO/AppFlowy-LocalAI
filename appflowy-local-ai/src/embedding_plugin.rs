use crate::embedding_ops::EmbeddingPluginOperation;
use std::collections::HashMap;

use crate::state::LLMState;
use anyhow::anyhow;
use anyhow::Result;
use appflowy_plugin::core::plugin::{Plugin, PluginInfo, RunningState, RunningStateSender};
use appflowy_plugin::error::PluginError;
use appflowy_plugin::manager::PluginManager;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::{info, trace};

pub struct LocalEmbedding {
  plugin_manager: Arc<PluginManager>,
  plugin_config: RwLock<Option<EmbeddingPluginConfig>>,
  state_notify: tokio::sync::broadcast::Sender<LLMState>,
  state: RwLock<LLMState>,
  running_state_sender: RunningStateSender,
}

impl LocalEmbedding {
  pub fn new(plugin_manager: Arc<PluginManager>) -> Self {
    let running_state_sender = tokio::sync::broadcast::channel(10).0;
    let (state_notify, _) = tokio::sync::broadcast::channel(10);
    Self {
      plugin_manager,
      plugin_config: Default::default(),
      state: RwLock::new(LLMState::Loading),
      state_notify,
      running_state_sender,
    }
  }

  pub async fn init_embedding_plugin(
    &self,
    config: EmbeddingPluginConfig,
  ) -> Result<(), PluginError> {
    if let Some(existing_config) = self.plugin_config.read().await.as_ref() {
      trace!(
        "[Embedding Plugin] existing config: {:?}, new config:{:?}",
        existing_config,
        config
      );
    }

    let info = PluginInfo {
      name: "embedding".to_string(),
      exec_path: config.bin_path,
    };
    self.update_state(LLMState::Loading).await;
    let plugin_id = self
      .plugin_manager
      .create_plugin(info, self.running_state_sender.clone())
      .await?;

    let mut params = json!({
        "absolute_model_path":config.model_path,
    });

    if let Some(persist_directory) = config.persist_directory {
      params["persist_directory"] = json!(persist_directory);
    }

    let plugin = self.plugin_manager.init_plugin(plugin_id, params).await?;
    info!("[Embedding Plugin] {} setup success", plugin);
    self.update_state(LLMState::Ready { plugin_id }).await;
    Ok(())
  }

  pub fn subscribe_running_state(&self) -> tokio::sync::broadcast::Receiver<RunningState> {
    self.running_state_sender.subscribe()
  }

  pub async fn generate_embedding(&self, text: &str) -> Result<Vec<Vec<f64>>, PluginError> {
    trace!("[Embedding Plugin] generate embedding for text: {}", text);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let embeddings = operation.embed_documents(text).await?;
    Ok(embeddings)
  }

  pub async fn index(
    &self,
    text: &str,
    metadata: HashMap<String, Value>,
  ) -> Result<(), PluginError> {
    trace!("[Embedding Plugin] generate embedding for text: {}", text);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    operation.index_document(text, metadata).await?;
    Ok(())
  }

  pub async fn similarity_search(
    &self,
    query: &str,
    filter: HashMap<String, Value>,
  ) -> Result<Vec<String>, PluginError> {
    trace!("[Embedding Plugin] similarity search for query: {}", query);
    self.wait_plugin_ready().await?;
    let plugin = self.get_embedding_plugin().await?;
    let operation = EmbeddingPluginOperation::new(plugin);
    let result = operation.similarity_search(query, filter).await?;
    Ok(result)
  }

  async fn get_embedding_plugin(&self) -> Result<Weak<Plugin>> {
    let plugin_id = self.state.read().await.plugin_id()?;
    let plugin = self.plugin_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }

  async fn update_state(&self, state: LLMState) {
    *self.state.write().await = state.clone();
  }

  async fn wait_plugin_ready(&self) -> Result<()> {
    let is_loading = self.state.read().await.is_loading();
    if !is_loading {
      return Ok(());
    }
    info!("[Embedding Plugin] wait for plugin to be ready");
    let mut rx = self.state_notify.subscribe();
    let timeout_duration = Duration::from_secs(30);
    let result = timeout(timeout_duration, async {
      while let Ok(state) = rx.recv().await {
        if state.is_ready() {
          break;
        }
      }
    })
    .await;

    match result {
      Ok(_) => {
        trace!("[Embedding Plugin] is ready");
        Ok(())
      },
      Err(_) => Err(anyhow!("Timeout while waiting for chat plugin to be ready")),
    }
  }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EmbeddingPluginConfig {
  pub bin_path: PathBuf,
  pub model_path: PathBuf,
  pub persist_directory: Option<PathBuf>,
}

impl EmbeddingPluginConfig {
  pub fn new<T: Into<PathBuf>>(
    bin_path: T,
    model_path: T,
    storage_path: Option<PathBuf>,
  ) -> Result<Self> {
    let bin_path = bin_path.into();
    let model_path = model_path.into();
    if !bin_path.exists() {
      return Err(anyhow!(
        "Embedding binary path does not exist: {:?}",
        bin_path
      ));
    }
    if !bin_path.is_file() {
      return Err(anyhow!(
        "Embedding binary path is not a file: {:?}",
        bin_path
      ));
    }

    // Check if local_model_dir exists and is a directory
    if !model_path.exists() {
      return Err(anyhow!("embedding model does not exist: {:?}", model_path));
    }
    if !model_path.is_file() {
      return Err(anyhow!("embedding model is not a file: {:?}", model_path));
    }

    Ok(Self {
      bin_path,
      model_path,
      persist_directory: storage_path,
    })
  }
}
