use crate::config::AppConfig;

pub type AudioSample = f32;

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<AudioSample>,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

pub fn generate_dummy_audio(config: &AppConfig, duration_ms: u32) -> AudioChunk {
    let total_samples = (config.sample_rate_hz as usize * duration_ms as usize) / 1_000;
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        let phase = (i % 100) as f32 / 100.0;
        let sample = if i % 2 == 0 { 0.02 * phase } else { 0.0 };
        samples.push(sample);
    }

    AudioChunk {
        samples,
        sample_rate_hz: config.sample_rate_hz,
        channels: config.channels,
    }
}
