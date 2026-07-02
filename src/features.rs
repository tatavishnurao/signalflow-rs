use crate::{
    mel::{apply_mel_filterbank, build_mel_filterbank, MelConfig},
    spectrum::power_spectrum,
    window::{window_frame, WindowFunction},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogMelConfig {
    pub sample_rate_hz: u32,
    pub fft_size: usize,
    pub num_mel_bins: usize,
    pub min_freq_hz: f32,
    pub max_freq_hz: f32,
    pub epsilon: f32,
}

impl LogMelConfig {
    pub fn speech_default(sample_rate_hz: u32, fft_size: usize) -> Self {
        Self {
            sample_rate_hz,
            fft_size,
            num_mel_bins: 40,
            min_freq_hz: 0.0,
            max_freq_hz: sample_rate_hz as f32 / 2.0,
            epsilon: 1e-6,
        }
    }
}

pub fn log_compress(values: &[f32], epsilon: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let epsilon = if epsilon <= 0.0 { 1e-6 } else { epsilon };
    values
        .iter()
        .map(|&value| (value.max(0.0) + epsilon).ln())
        .collect()
}

pub fn log_mel_frame(frame: &[f32], config: LogMelConfig) -> Vec<f32> {
    if frame.is_empty() {
        return Vec::new();
    }

    let windowed = window_frame(frame, WindowFunction::Hann);
    let power = power_spectrum(&windowed);
    let filterbank = build_mel_filterbank(MelConfig {
        sample_rate_hz: config.sample_rate_hz,
        fft_size: config.fft_size,
        num_mel_bins: config.num_mel_bins,
        min_freq_hz: config.min_freq_hz,
        max_freq_hz: config.max_freq_hz,
    });

    if filterbank.is_empty() {
        return Vec::new();
    }

    let mel_energies = apply_mel_filterbank(&power, &filterbank);
    log_compress(&mel_energies, config.epsilon)
}

pub fn log_mel_features(frames: &[Vec<f32>], config: LogMelConfig) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }

    frames
        .iter()
        .map(|frame| log_mel_frame(frame, config))
        .filter(|row| !row.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{log_compress, log_mel_features, log_mel_frame, LogMelConfig};

    #[test]
    fn log_compress_empty_input() {
        assert!(log_compress(&[], 1e-6).is_empty());
    }

    #[test]
    fn log_compress_handles_zero_values() {
        let values = log_compress(&[0.0, 1.0], 1e-6);

        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
    }

    #[test]
    fn log_compress_clamps_negative_values() {
        let values = log_compress(&[-1.0, 0.5], 1e-6);

        assert!(values[0].is_finite());
        assert!(values[1].is_finite());
    }

    #[test]
    fn log_mel_frame_empty_input() {
        assert!(log_mel_frame(&[], LogMelConfig::speech_default(16_000, 400)).is_empty());
    }

    #[test]
    fn log_mel_frame_returns_num_mel_bins() {
        let frame = vec![1.0; 400];
        let features = log_mel_frame(&frame, LogMelConfig::speech_default(16_000, 400));

        assert_eq!(features.len(), 40);
    }

    #[test]
    fn log_mel_features_empty_frames() {
        assert!(log_mel_features(&[], LogMelConfig::speech_default(16_000, 400)).is_empty());
    }

    #[test]
    fn log_mel_features_returns_expected_shape() {
        let frames = vec![vec![1.0; 400]; 8];
        let features = log_mel_features(&frames, LogMelConfig::speech_default(16_000, 400));

        assert_eq!(features.len(), 8);
        assert!(features.iter().all(|row| row.len() == 40));
    }

    #[test]
    fn log_mel_features_values_are_finite() {
        let frames = vec![vec![1.0; 400]; 8];
        let features = log_mel_features(&frames, LogMelConfig::speech_default(16_000, 400));

        assert!(features.iter().flatten().all(|value| value.is_finite()));
    }
}
