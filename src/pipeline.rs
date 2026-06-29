use anyhow::Result;

use crate::{
    audio::generate_dummy_audio,
    config::AppConfig,
    dsp::{normalize_samples, rms_energy},
};

#[derive(Debug, Clone, Copy)]
pub struct PipelineReport {
    pub num_samples: usize,
    pub frame_size_samples: usize,
    pub hop_size_samples: usize,
    pub rms_energy: f32,
}

pub fn run_dummy_pipeline(config: &AppConfig) -> Result<PipelineReport> {
    let mut chunk = generate_dummy_audio(config, 100);
    normalize_samples(&mut chunk.samples);

    Ok(PipelineReport {
        num_samples: chunk.samples.len(),
        frame_size_samples: config.frame_size_samples(),
        hop_size_samples: config.hop_size_samples(),
        rms_energy: rms_energy(&chunk.samples),
    })
}
