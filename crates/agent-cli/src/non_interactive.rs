use agent_model::{LlmMessage, LlmRole, MessagePart, TextPart};
use agent_core::Agent;
use agent_session::SessionStore;
use secrecy::SecretString;

use crate::compaction::maybe_auto_compact;
use crate::runtime::AppBackend;

pub(crate) struct NonInteractiveConfig<'a> {
    pub prompt: String,
    pub api_key: &'a str,
    pub auto_compact_threshold: usize,
    pub auto_compact_keep_last: usize,
    pub backend: &'a AppBackend,
    pub summary_model: &'a str,
}

pub(crate) async fn run_prompt(
    session: &mut SessionStore,
    agent: &mut Agent<AppBackend>,
    config: NonInteractiveConfig<'_>,
) -> anyhow::Result<()> {
    agent.set_state(agent_core::AgentState {
        messages: session.messages(),
    });
    let run = agent
        .prompt(
            LlmMessage {
                role: LlmRole::User,
                parts: vec![MessagePart::Text(TextPart {
                    text: config.prompt,
                })],
            },
            SecretString::new(config.api_key.to_string().into_boxed_str()),
        )
        .await?;
    print_run_output(&run);
    session.append_run(&run)?;
    if let Some(compaction) = maybe_auto_compact(
        session,
        config.auto_compact_threshold,
        config.auto_compact_keep_last,
        Some(config.api_key.to_string()),
        config.backend,
        config.summary_model,
    )
    .await?
    {
        println!(
            "[system] auto-compacted {} messages into {}",
            compaction.compacted_message_count,
            compaction.summary_entry_id
        );
    }
    Ok(())
}

fn print_run_output(run: &agent_core::AgentRunResult) {
    for message in &run.new_messages {
        let role = match message {
            agent_core::AgentMessage::User(_) => "user",
            agent_core::AgentMessage::Assistant(_) => "assistant",
            agent_core::AgentMessage::ToolResult(_) => "tool",
        };
        println!("[{role}]");
        for part in &message.as_llm_message().parts {
            match part {
                MessagePart::Text(text) => println!("{}", text.text),
                MessagePart::ToolCall(call) => {
                    println!("tool_call {} {}", call.name, call.arguments_json);
                }
                MessagePart::ToolResult(result) => {
                    println!("tool_result {} {}", result.call_id, result.content);
                }
            }
        }
    }
}
