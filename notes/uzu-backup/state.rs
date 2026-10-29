use std::sync::Arc;

use tokio::sync::Mutex;
use uzu::{session::chat::ChatSession, types::basic::ReasoningEffort};

use crate::common::thinking::ThinkingSupport;

pub struct ServerState {
    pub model_name: String,
    pub session: Arc<Mutex<ChatSession>>,
    pub thinking_support: ThinkingSupport,
    pub prefix_cache: bool,
    /// Applied when the request omits both `reasoning_effort` and
    /// `enable_thinking`. Set through `UZU_REASONING_EFFORT`.
    pub default_reasoning: Option<ReasoningEffort>,
}
