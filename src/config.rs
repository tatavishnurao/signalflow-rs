#[derive(Debug, Clone, Copy)]
pub struct AppConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_ms: u32,
    pub hop_ms: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 16_000,
            channels: 1,
            frame_ms: 25,
            hop_ms: 10,
        }
    }
}

impl AppConfig {
    pub fn frame_size_samples(&self) -> usize {
        (self.sample_rate_hz as usize * self.frame_ms as usize) / 1_000
    }

    pub fn hop_size_samples(&self) -> usize {
        (self.sample_rate_hz as usize * self.hop_ms as usize) / 1_000
    }
}

#[cfg(test)]
mod tests {
    use super::AppConfig;

    #[test]
    fn default_config_has_expected_values() {
        let config = AppConfig::default();

        assert_eq!(config.sample_rate_hz, 16_000);
        assert_eq!(config.channels, 1);
        assert_eq!(config.frame_ms, 25);
        assert_eq!(config.hop_ms, 10);
    }

    #[test]
    fn frame_size_samples_is_400() {
        assert_eq!(AppConfig::default().frame_size_samples(), 400);
    }

    #[test]
    fn hop_size_samples_is_160() {
        assert_eq!(AppConfig::default().hop_size_samples(), 160);
    }
}
