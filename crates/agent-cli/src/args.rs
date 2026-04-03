use crate::config::AppConfig;
use crate::paths::AppPaths;
use crate::provider_registry::Provider;

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
    println!("sessions_dir={}", config.resolved_session_dir(paths).display());
}
