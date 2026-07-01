use anyhow::Result;

use crate::{
    audio::generate_dummy_audio,
    config::AppConfig,
    dsp::{normalize_samples, rms_energy},
    framing::{frame_signal, FrameConfig},
    spectrum::magnitude_spectrum,
    window::{window_frame, WindowFunction},
};

#[derive(Debug, Clone, Copy)]
pub struct PipelineReport {
    pub num_samples: usize,
    pub frame_size_samples: usize,
    pub hop_size_samples: usize,
    pub num_frames: usize,
    pub first_windowed_frame_rms: f32,
    pub first_spectrum_bins: usize,
    pub first_spectrum_peak: f32,
    pub rms_energy: f32,
}

pub fn run_dummy_pipeline(config: &AppConfig) -> Result<PipelineReport> {
    let mut chunk = generate_dummy_audio(config, 100);
    normalize_samples(&mut chunk.samples);
    let frame_config = FrameConfig::new(config.frame_size_samples(), config.hop_size_samples());
    let frames = frame_signal(&chunk.samples, frame_config);
    let first_windowed_frame_rms = frames
        .first()
        .map(|frame| {
            let windowed = window_frame(frame, WindowFunction::Hann);
            rms_energy(&windowed)
        })
        .unwrap_or(0.0);
    let (first_spectrum_bins, first_spectrum_peak) = frames
        .first()
        .map(|frame| {
            let windowed = window_frame(frame, WindowFunction::Hann);
            let spectrum = magnitude_spectrum(&windowed);
            let peak = spectrum.iter().copied().fold(0.0, f32::max);
            (spectrum.len(), peak)
        })
        .unwrap_or((0, 0.0));

    Ok(PipelineReport {
        num_samples: chunk.samples.len(),
        frame_size_samples: frame_config.frame_size_samples,
        hop_size_samples: frame_config.hop_size_samples,
        num_frames: frames.len(),
        first_windowed_frame_rms,
        first_spectrum_bins,
        first_spectrum_peak,
        rms_energy: rms_energy(&chunk.samples),
    })
}

#[cfg(test)]
mod tests {
    use super::run_dummy_pipeline;
    use crate::config::AppConfig;

    #[test]
    fn dummy_pipeline_succeeds() {
        assert!(run_dummy_pipeline(&AppConfig::default()).is_ok());
    }

    #[test]
    fn dummy_pipeline_report_has_expected_values() {
        let report = run_dummy_pipeline(&AppConfig::default()).unwrap();

        assert_eq!(report.num_samples, 1_600);
        assert_eq!(report.frame_size_samples, 400);
        assert_eq!(report.hop_size_samples, 160);
        assert_eq!(report.num_frames, 8);
        assert!(report.first_windowed_frame_rms >= 0.0);
        assert_eq!(report.first_spectrum_bins, 201);
        assert!(report.first_spectrum_peak >= 0.0);
    }
}
