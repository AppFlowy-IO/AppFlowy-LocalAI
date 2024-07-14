use crate::chat_ops::ChatPluginOperation;
use anyhow::{anyhow, Result};
use appflowy_plugin::core::plugin::{
  Plugin, PluginInfo, RunningState, RunningStateReceiver, RunningStateSender,
};
use appflowy_plugin::error::PluginError;
use appflowy_plugin::manager::PluginManager;
use appflowy_plugin::util::{get_operating_system, OperatingSystem};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::io;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tokio_stream::wrappers::{ReceiverStream, WatchStream};
use tokio_stream::StreamExt;
use tracing::{error, info, instrument, trace};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalLLMSetting {
  pub chat_bin_path: String,
  pub chat_model_path: String,
  pub embedding_model_path: String,
  pub enabled: bool,
}

impl LocalLLMSetting {
  pub fn validate(&self) -> Result<()> {
    ChatPluginConfig::new(&self.chat_bin_path, &self.chat_model_path)?;
    Ok(())
  }
}

pub struct LocalChatLLMChat {
  plugin_manager: Arc<PluginManager>,
  plugin_config: RwLock<Option<ChatPluginConfig>>,
  running_state: RunningStateSender,
  #[allow(dead_code)]
  // keep at least one receiver that make sure the sender can receive value
  running_state_rx: RunningStateReceiver,
}

impl LocalChatLLMChat {
  pub fn new(plugin_manager: Arc<PluginManager>) -> Self {
    let (running_state, rx) = tokio::sync::watch::channel(RunningState::Connecting);
    Self {
      plugin_manager,
      plugin_config: Default::default(),
      running_state: Arc::new(running_state),
      running_state_rx: rx,
    }
  }

  /// Creates a new chat session.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  pub async fn create_chat(&self, chat_id: &str) -> Result<(), PluginError> {
    trace!("[Chat Plugin] create chat: {}", chat_id);
    self.wait_until_plugin_ready().await?;

    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    operation.create_chat(chat_id, true).await?;
    Ok(())
  }

  /// Closes an existing chat session.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session to close.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  pub async fn close_chat(&self, chat_id: &str) -> Result<()> {
    trace!("[Chat Plugin] close chat: {}", chat_id);
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    operation.close_chat(chat_id).await?;
    Ok(())
  }

  pub fn subscribe_running_state(&self) -> WatchStream<RunningState> {
    WatchStream::new(self.running_state.subscribe())
  }

  /// Asks a question and returns a stream of responses.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  /// * `message` - A string slice containing the question or message to send.
  ///
  /// # Returns
  ///
  /// A `Result<ReceiverStream<anyhow::Result<Bytes, SidecarError>>>` containing a stream of responses.
  pub async fn stream_question(
    &self,
    chat_id: &str,
    message: &str,
  ) -> Result<ReceiverStream<anyhow::Result<Bytes, PluginError>>, PluginError> {
    trace!("[Chat Plugin] ask question: {}", message);
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    let stream = operation.stream_message(chat_id, message, true).await?;
    Ok(stream)
  }

  pub async fn get_related_question(&self, chat_id: &str) -> Result<Vec<String>, PluginError> {
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    let values = operation.get_related_questions(chat_id).await?;
    Ok(values)
  }

  pub async fn index_file(&self, chat_id: &str, file_path: PathBuf) -> Result<(), PluginError> {
    if !file_path.exists() {
      return Err(PluginError::Io(io::Error::new(
        io::ErrorKind::NotFound,
        "file not found",
      )));
    }

    let file_path = file_path.to_str().ok_or(PluginError::Io(io::Error::new(
      io::ErrorKind::NotFound,
      "file path invalid",
    )))?;

    self.wait_until_plugin_ready().await?;
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    trace!("[Chat Plugin] indexing file: {}", file_path);
    operation.index_file(chat_id, file_path).await?;
    Ok(())
  }

  /// Generates a complete answer for a given message.
  ///
  /// # Arguments
  ///
  /// * `chat_id` - A string slice containing the unique identifier for the chat session.
  /// * `message` - A string slice containing the message to generate an answer for.
  ///
  /// # Returns
  ///
  /// A `Result<String>` containing the generated answer.
  pub async fn ask_question(&self, chat_id: &str, message: &str) -> Result<String, PluginError> {
    self.wait_until_plugin_ready().await?;
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    let answer = operation.send_message(chat_id, message, true).await?;
    Ok(answer)
  }

  #[instrument(skip_all, err)]
  pub async fn destroy_chat_plugin(&self) -> Result<()> {
    let plugin_id = self.running_state.borrow().plugin_id();
    if let Some(plugin_id) = plugin_id {
      if let Err(err) = self.plugin_manager.remove_plugin(plugin_id).await {
        error!("remove plugin failed: {:?}", err);
      }
    }

    Ok(())
  }

  #[instrument(skip_all, err)]
  pub async fn init_chat_plugin(&self, config: ChatPluginConfig) -> Result<()> {
    let state = self.running_state.borrow().clone();
    if state.is_ready() {
      if let Some(existing_config) = self.plugin_config.read().await.as_ref() {
        trace!(
          "[Chat Plugin] existing config: {:?}, new config:{:?}",
          existing_config,
          config
        );
      }
    }

    let system = get_operating_system();
    // Initialize chat plugin if the config is different
    // If the chat_bin_path is different, remove the old plugin
    if let Err(err) = self.destroy_chat_plugin().await {
      error!("[Chat Plugin] failed to destroy plugin: {:?}", err);
    }

    // create new plugin
    trace!("[Chat Plugin] create chat plugin: {:?}", config);
    let plugin_info = PluginInfo {
      name: "chat_plugin".to_string(),
      exec_path: config.chat_bin_path.clone(),
    };
    let plugin_id = self
      .plugin_manager
      .create_plugin(plugin_info, self.running_state.clone())
      .await?;

    // init plugin
    trace!("[Chat Plugin] init chat plugin model: {:?}", plugin_id);
    let model_path = config.chat_model_path.clone();
    let mut params = match system {
      OperatingSystem::Windows => {
        let device = config.device.as_str();
        serde_json::json!({
          "absolute_chat_model_path": model_path,
          "device": device,
        })
      },
      OperatingSystem::Linux => {
        let device = config.device.as_str();
        serde_json::json!({
          "absolute_chat_model_path": model_path,
          "device": device,
        })
      },
      OperatingSystem::MacOS => {
        let device = config.device.as_str();
        serde_json::json!({
          "absolute_chat_model_path": model_path,
          "device": device,
        })
      },
      _ => {
        return Err(anyhow!("Unsupported operating system"));
      },
    };

    params["verbose"] = serde_json::json!(config.verbose);
    if let Some(related_model_path) = config.related_model_path.clone() {
      params["absolute_related_model_path"] = serde_json::json!(related_model_path);
    }

    if let (Some(embedding_model_path), Some(persist_directory)) = (
      config.embedding_model_path.clone(),
      config.persist_directory.clone(),
    ) {
      params["vectorstore_config"] = serde_json::json!({
        "absolute_model_path": embedding_model_path,
        "persist_directory": persist_directory,
      });
    }

    info!(
      "[Chat Plugin] setup chat plugin: {:?}, params: {:?}",
      plugin_id, params
    );
    let plugin = self.plugin_manager.init_plugin(plugin_id, params).await?;
    info!("[Chat Plugin] {} setup success", plugin);
    self.plugin_config.write().await.replace(config);
    Ok(())
  }

  /// Waits for the plugin to be ready.
  ///
  /// The wait_plugin_ready method is an asynchronous function designed to ensure that the chat
  /// plugin is in a ready state before allowing further operations. This is crucial for maintaining
  /// the correct sequence of operations and preventing errors that could occur if operations are
  /// attempted on an unready plugin.
  ///
  /// # Returns
  ///
  /// A `Result<()>` indicating success or failure.
  async fn wait_until_plugin_ready(&self) -> Result<()> {
    let is_loading = self.running_state.borrow().is_loading();
    if !is_loading {
      return Ok(());
    }
    info!("[Chat Plugin] wait for chat plugin to be ready");
    let mut rx = self.subscribe_running_state();
    let timeout_duration = Duration::from_secs(30);
    let result = timeout(timeout_duration, async {
      while let Some(state) = rx.next().await {
        if state.is_ready() {
          break;
        }
      }
    })
    .await;

    match result {
      Ok(_) => {
        trace!("[Chat Plugin] is ready");
        Ok(())
      },
      Err(_) => Err(anyhow!("Timeout while waiting for chat plugin to be ready")),
    }
  }

  /// Retrieves the chat plugin.
  ///
  /// # Returns
  ///
  /// A `Result<Weak<Plugin>>` containing a weak reference to the plugin.
  async fn get_chat_plugin(&self) -> Result<Weak<Plugin>, PluginError> {
    let plugin_id = self
      .running_state
      .borrow()
      .plugin_id()
      .ok_or_else(|| PluginError::Internal(anyhow!("chat plugin not initialized")))?;
    let plugin = self.plugin_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ChatPluginConfig {
  pub chat_bin_path: PathBuf,
  pub chat_model_path: PathBuf,
  pub related_model_path: Option<PathBuf>,
  pub embedding_model_path: Option<PathBuf>,
  pub persist_directory: Option<PathBuf>,
  pub device: String,
  pub verbose: bool,
}

impl ChatPluginConfig {
  pub fn new<T: Into<PathBuf>>(chat_bin_path: T, chat_model_path: T) -> Result<Self> {
    let chat_bin_path = chat_bin_path.into();
    if !chat_bin_path.exists() {
      return Err(anyhow!(
        "Chat binary path does not exist: {:?}",
        chat_bin_path
      ));
    }
    if !chat_bin_path.is_file() {
      return Err(anyhow!(
        "Chat binary path is not a file: {:?}",
        chat_bin_path
      ));
    }

    // Check if local_model_dir exists and is a directory
    let chat_model_path = chat_model_path.into();
    if !chat_model_path.exists() {
      return Err(anyhow!("Local model does not exist: {:?}", chat_model_path));
    }
    if !chat_model_path.is_file() {
      return Err(anyhow!("Local model is not a file: {:?}", chat_model_path));
    }

    Ok(Self {
      chat_bin_path,
      chat_model_path,
      related_model_path: None,
      embedding_model_path: None,
      persist_directory: None,
      device: "cpu".to_string(),
      verbose: false,
    })
  }

  pub fn with_device(mut self, device: &str) -> Self {
    self.device = device.to_string();
    self
  }

  pub fn with_verbose(mut self, verbose: bool) -> Self {
    self.verbose = verbose;
    self
  }
  pub fn set_rag_enabled(
    &mut self,
    embedding_model_path: &PathBuf,
    persist_directory: &PathBuf,
  ) -> Result<()> {
    if !embedding_model_path.exists() {
      return Err(anyhow!(
        "embedding model path does not exist: {:?}",
        embedding_model_path
      ));
    }
    if !embedding_model_path.is_file() {
      return Err(anyhow!(
        "embedding model is not a file: {:?}",
        embedding_model_path
      ));
    }

    if !persist_directory.exists() {
      std::fs::create_dir_all(persist_directory)?;
    }

    self.embedding_model_path = Some(embedding_model_path.clone());
    self.persist_directory = Some(persist_directory.clone());
    Ok(())
  }

  pub fn with_related_model_path<T: Into<PathBuf>>(mut self, related_model_path: T) -> Self {
    self.related_model_path = Some(related_model_path.into());
    self
  }
}
