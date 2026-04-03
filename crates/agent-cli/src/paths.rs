use std::path::PathBuf;

use anyhow::Context;

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub state_dir: PathBuf,
    pub sessions_dir: PathBuf,
}

impl AppPaths {
    pub fn discover() -> anyhow::Result<Self> {
        let config_file = std::env::var_os("MINIATURE_AGENT_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| xdg_config_home().join("miniature-agent").join("config.toml"));
        let config_dir = config_file
            .parent()
            .map(PathBuf::from)
            .context("config file path has no parent directory")?;

        let state_root = std::env::var_os("MINIATURE_AGENT_STATE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| xdg_state_home().join("miniature-agent"));
        let sessions_dir = state_root.join("sessions");

        Ok(Self {
            config_dir,
            config_file,
            state_dir: state_root,
            sessions_dir,
        })
    }

    pub fn ensure_directories(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.config_dir)
            .with_context(|| format!("failed to create {}", self.config_dir.display()))?;
        std::fs::create_dir_all(&self.sessions_dir)
            .with_context(|| format!("failed to create {}", self.sessions_dir.display()))?;
        Ok(())
    }
}

fn xdg_config_home() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".config"))
}

fn xdg_state_home() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local").join("state"))
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_standard_xdg_paths_from_env() {
        unsafe {
            std::env::set_var("XDG_CONFIG_HOME", "/tmp/miniature-agent-config");
            std::env::set_var("XDG_STATE_HOME", "/tmp/miniature-agent-state");
            std::env::remove_var("MINIATURE_AGENT_CONFIG");
            std::env::remove_var("MINIATURE_AGENT_STATE_DIR");
        }

        let paths = AppPaths::discover().unwrap();
        assert_eq!(
            paths.config_file,
            PathBuf::from("/tmp/miniature-agent-config/miniature-agent/config.toml")
        );
        assert_eq!(
            paths.sessions_dir,
            PathBuf::from("/tmp/miniature-agent-state/miniature-agent/sessions")
        );
    }
}
