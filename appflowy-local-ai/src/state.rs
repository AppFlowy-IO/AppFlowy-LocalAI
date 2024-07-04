use anyhow::anyhow;
use appflowy_plugin::core::plugin::PluginId;
use appflowy_plugin::util::get_operating_system;

#[derive(Debug, Clone)]
pub enum LLMState {
    Uninitialized,
    Loading,
    Ready { plugin_id: PluginId },
}

impl LLMState {
    pub(crate) fn plugin_id(&self) -> anyhow::Result<PluginId> {
        match self {
            LLMState::Ready { plugin_id } => Ok(*plugin_id),
            _ => Err(anyhow!("chat plugin is not ready")),
        }
    }

    pub(crate) fn is_loading(&self) -> bool {
        matches!(self, LLMState::Loading)
    }

    #[allow(dead_code)]
    fn is_uninitialized(&self) -> bool {
        matches!(self, LLMState::Uninitialized)
    }

    pub(crate) fn is_ready(&self) -> bool {
        let system = get_operating_system();
        if system.is_desktop() {
            matches!(self, LLMState::Ready { .. })
        } else {
            false
        }
    }
}
