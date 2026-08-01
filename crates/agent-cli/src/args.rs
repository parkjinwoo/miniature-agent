use crate::config::AppConfig;
use crate::paths::AppPaths;
use crate::provider_registry::Provider;

pub(crate) fn validate(args: &[String]) -> anyhow::Result<()> {
    let mut index = 1usize;
    while index < args.len() {
        let arg = &args[index];
        match arg.as_str() {
            "--help"
            | "-h"
            | "--version"
            | "-V"
            | "--compact"
            | "--print-paths"
            | "--write-default-config"
            | "--list-sessions"
            | "--new-session"
            | "--full-access" => {}
            "--provider" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("--provider requires a value"))?;
                anyhow::ensure!(
                    Provider::parse(value).is_some(),
                    "unknown provider {value:?}; expected openai, anthropic, or compatible"
                );
            }
            "--model" | "--prompt" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or_else(|| anyhow::anyhow!("{arg} requires a value"))?;
                anyhow::ensure!(!value.is_empty(), "{arg} requires a non-empty value");
            }
            _ if arg.starts_with("--provider=") => {
                let value = arg.trim_start_matches("--provider=");
                anyhow::ensure!(
                    Provider::parse(value).is_some(),
                    "unknown provider {value:?}; expected openai, anthropic, or compatible"
                );
            }
            _ if arg.starts_with("--model=") || arg.starts_with("--prompt=") => {
                let (_, value) = arg.split_once('=').unwrap_or_default();
                anyhow::ensure!(
                    !value.is_empty(),
                    "{} requires a non-empty value",
                    &arg[..arg.find('=').unwrap_or(arg.len())]
                );
            }
            _ => anyhow::bail!("unknown argument {arg:?}; use --help for usage"),
        }
        index += 1;
    }
    anyhow::ensure!(
        !args.iter().any(|arg| arg == "--full-access") || parse_prompt(args).is_some(),
        "--full-access only applies to non-interactive --prompt mode"
    );
    Ok(())
}

pub(crate) fn print_help() {
    println!(
        "miniature-agent {}\n\nUSAGE:\n    miniature-agent [OPTIONS]\n\nOPTIONS:\n    --provider <openai|anthropic|compatible>\n    --model <MODEL>             Override the provider's default model\n    --prompt <TEXT>             Run one non-interactive prompt\n    --full-access              Allow write, edit, and bash in prompt mode\n    --new-session              Start a fresh session\n    --compact                  Compact the current session before running\n    --list-sessions            List saved sessions\n    --print-paths              Print resolved config and state paths\n    --write-default-config     Write the example config\n    -h, --help                 Print help\n    -V, --version              Print version",
        env!("CARGO_PKG_VERSION")
    );
}

pub(crate) fn parse_provider(args: &[String]) -> Option<Provider> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        let value = if let Some(value) = arg.strip_prefix("--provider=") {
            Some(value)
        } else if arg == "--provider" {
            iter.next().map(String::as_str)
        } else {
            None
        };

        if let Some(provider) = value.and_then(Provider::parse) {
            return Some(provider);
        }
    }
    None
}

pub(crate) fn parse_model(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--model=") {
            return Some(value.to_string());
        }
        if arg == "--model" {
            return iter.next().cloned();
        }
    }
    None
}

pub(crate) fn parse_prompt(args: &[String]) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--prompt=") {
            return Some(value.to_string());
        }
        if arg == "--prompt" {
            return iter.next().cloned();
        }
    }
    None
}

pub(crate) fn print_paths(paths: &AppPaths, config: &AppConfig) {
    println!("config_file={}", paths.config_file.display());
    println!("config_dir={}", paths.config_dir.display());
    println!("state_dir={}", paths.state_dir.display());
    println!(
        "sessions_dir={}",
        config.resolved_session_dir(paths).display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_rejects_unknown_providers_and_missing_values() {
        let unknown = vec![
            "miniature-agent".to_string(),
            "--provider".to_string(),
            "mystery".to_string(),
        ];
        assert!(
            validate(&unknown)
                .unwrap_err()
                .to_string()
                .contains("unknown provider")
        );

        let missing = vec!["miniature-agent".to_string(), "--prompt".to_string()];
        assert!(
            validate(&missing)
                .unwrap_err()
                .to_string()
                .contains("requires a value")
        );
    }

    #[test]
    fn validation_accepts_the_documented_flags() {
        let args = vec![
            "miniature-agent".to_string(),
            "--provider=compatible".to_string(),
            "--model".to_string(),
            "local-model".to_string(),
            "--prompt=hello".to_string(),
            "--new-session".to_string(),
        ];
        validate(&args).unwrap();
    }

    #[test]
    fn full_access_requires_non_interactive_prompt_mode() {
        let args = vec!["miniature-agent".to_string(), "--full-access".to_string()];
        assert!(
            validate(&args)
                .unwrap_err()
                .to_string()
                .contains("only applies")
        );
    }
}
