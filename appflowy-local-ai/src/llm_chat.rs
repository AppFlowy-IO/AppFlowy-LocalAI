use crate::chat_plugin::ChatPluginOperation;
use crate::state::LLMState;
use anyhow::{anyhow, Result};
use appflowy_plugin::core::plugin::{Plugin, PluginInfo};
use appflowy_plugin::error::SidecarError;
use appflowy_plugin::manager::SidecarManager;
use appflowy_plugin::util::{get_operating_system, OperatingSystem};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, instrument, trace};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LocalLLMSetting {
  pub chat_bin_path: String,
  pub chat_model_path: String,
  pub enabled: bool,
}

impl LocalLLMSetting {
  pub fn validate(&self) -> Result<()> {
    ChatPluginConfig::new(&self.chat_bin_path, &self.chat_model_path)?;
    Ok(())
  }
  pub fn chat_config(&self) -> Result<ChatPluginConfig> {
    let config = ChatPluginConfig::new(&self.chat_bin_path, &self.chat_model_path)?;
    Ok(config)
  }
}

pub struct LocalChatLLMChat {
  sidecar_manager: Arc<SidecarManager>,
  state: RwLock<LLMState>,
  state_notify: tokio::sync::broadcast::Sender<LLMState>,
  plugin_config: RwLock<Option<ChatPluginConfig>>,
}

impl LocalChatLLMChat {
  pub fn new(sidecar_manager: Arc<SidecarManager>) -> Self {
    let (state_notify, _) = tokio::sync::broadcast::channel(10);
    Self {
      sidecar_manager,
      state: RwLock::new(LLMState::Loading),
      state_notify,
      plugin_config: Default::default(),
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
  pub async fn create_chat(&self, chat_id: &str) -> Result<()> {
    trace!("[Chat Plugin] create chat: {}", chat_id);
    self.wait_plugin_ready().await?;

    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    operation.create_chat(chat_id).await?;
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
  pub async fn ask_question(
    &self,
    chat_id: &str,
    message: &str,
  ) -> Result<ReceiverStream<anyhow::Result<Bytes, SidecarError>>> {
    trace!("[Chat Plugin] ask question: {}", message);
    self.wait_plugin_ready().await?;
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    let stream = operation.stream_message(chat_id, message).await?;
    Ok(stream)
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
  pub async fn generate_answer(&self, chat_id: &str, message: &str) -> Result<String> {
    let plugin = self.get_chat_plugin().await?;
    let operation = ChatPluginOperation::new(plugin);
    let answer = operation.send_message(chat_id, message).await?;
    Ok(answer)
  }

  #[instrument(skip_all, err)]
  pub async fn destroy_chat_plugin(&self) -> Result<()> {
    if let Ok(plugin_id) = self.state.read().await.plugin_id() {
      if let Err(err) = self.sidecar_manager.remove_plugin(plugin_id).await {
        error!("remove plugin failed: {:?}", err);
      }
    }

    self.update_state(LLMState::Uninitialized).await;
    Ok(())
  }

  #[instrument(skip_all, err)]
  pub async fn init_chat_plugin(&self, config: ChatPluginConfig) -> Result<()> {
    if self.state.read().await.is_ready() {
      if let Some(existing_config) = self.plugin_config.read().await.as_ref() {
        if existing_config == &config {
          trace!("[Chat Plugin] already initialized with the same config");
          return Ok(());
        } else {
          trace!(
            "[Chat Plugin] existing config: {:?}, new config:{:?}",
            existing_config,
            config
          );
        }
      }
    }

    let system = get_operating_system();
    // Initialize chat plugin if the config is different
    // If the chat_bin_path is different, remove the old plugin
    if let Err(err) = self.destroy_chat_plugin().await {
      error!("[Chat Plugin] failed to destroy plugin: {:?}", err);
    }
    self.update_state(LLMState::Loading).await;

    // create new plugin
    trace!("[Chat Plugin] create chat plugin: {:?}", config);
    let plugin_info = PluginInfo {
      name: "chat_plugin".to_string(),
      exec_path: config.chat_bin_path.clone(),
    };
    let plugin_id = self.sidecar_manager.create_plugin(plugin_info).await?;

    // init plugin
    trace!("[Chat Plugin] init chat plugin model: {:?}", plugin_id);
    let model_path = config.chat_model_path.clone();
    let params = match system {
      OperatingSystem::Windows => {
        let device = config.device.as_str();
        serde_json::json!({
          "absolute_chat_model_path": model_path,
          // Currently, using GPU for windows will somehow cause windows to crash
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

    info!(
      "[Chat Plugin] setup chat plugin: {:?}, params: {:?}",
      plugin_id, params
    );
    let plugin = self.sidecar_manager.init_plugin(plugin_id, params)?;
    info!("[Chat Plugin] {} setup success", plugin);
    self.plugin_config.write().await.replace(config);
    self.update_state(LLMState::Ready { plugin_id }).await;
    Ok(())
  }

  async fn update_state(&self, state: LLMState) {
    *self.state.write().await = state.clone();
    let _ = self.state_notify.send(state);
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
  async fn wait_plugin_ready(&self) -> Result<()> {
    let is_loading = self.state.read().await.is_loading();
    if !is_loading {
      return Ok(());
    }
    info!("[Chat Plugin] wait for chat plugin to be ready");
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
  async fn get_chat_plugin(&self) -> Result<Weak<Plugin>> {
    let plugin_id = self.state.read().await.plugin_id()?;
    let plugin = self.sidecar_manager.get_plugin(plugin_id).await?;
    Ok(plugin)
  }
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct ChatPluginConfig {
  chat_bin_path: PathBuf,
  chat_model_path: PathBuf,
  device: String,
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
      device: "cpu".to_string(),
    })
  }

  pub fn with_device(mut self, device: &str) -> Self {
    self.device = device.to_string();
    self
  }
}
