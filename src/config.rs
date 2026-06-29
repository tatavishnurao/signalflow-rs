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
