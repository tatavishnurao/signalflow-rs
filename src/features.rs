use crate::{
    mel::{apply_mel_filterbank, build_mel_filterbank, MelConfig},
    window::hann_window,
};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{fmt, sync::Arc};

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

#[derive(Clone)]
pub struct LogMelProcessor {
    config: LogMelConfig,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
    fft: Arc<dyn Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>,
    power_buffer: Vec<f32>,
}

impl fmt::Debug for LogMelProcessor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LogMelProcessor")
            .field("config", &self.config)
            .field("window_len", &self.window.len())
            .field("filterbank_len", &self.filterbank.len())
            .finish()
    }
}

impl LogMelProcessor {
    pub fn new(config: LogMelConfig) -> Self {
        let fft_size = config.fft_size;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size.max(1));
        let filterbank = build_mel_filterbank(MelConfig {
            sample_rate_hz: config.sample_rate_hz,
            fft_size,
            num_mel_bins: config.num_mel_bins,
            min_freq_hz: config.min_freq_hz,
            max_freq_hz: config.max_freq_hz,
        });

        Self {
            config,
            window: hann_window(fft_size),
            filterbank,
            fft,
            fft_buffer: vec![Complex::new(0.0, 0.0); fft_size],
            power_buffer: vec![0.0; fft_size / 2 + 1],
        }
    }

    pub fn process_frame(&mut self, frame: &[f32]) -> Vec<f32> {
        if frame.len() != self.config.fft_size || frame.is_empty() || self.filterbank.is_empty() {
            return Vec::new();
        }

        for ((slot, sample), weight) in self
            .fft_buffer
            .iter_mut()
            .zip(frame.iter())
            .zip(self.window.iter())
        {
            *slot = Complex::new(sample * weight, 0.0);
        }
        self.fft.process(&mut self.fft_buffer);

        for (power, bin) in self.power_buffer.iter_mut().zip(self.fft_buffer.iter()) {
            *power = bin.re * bin.re + bin.im * bin.im;
        }

        let mel_energies = apply_mel_filterbank(&self.power_buffer, &self.filterbank);
        log_compress(&mel_energies, self.config.epsilon)
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
    LogMelProcessor::new(config).process_frame(frame)
}

pub fn log_mel_features(frames: &[Vec<f32>], config: LogMelConfig) -> Vec<Vec<f32>> {
    if frames.is_empty() {
        return Vec::new();
    }

    let mut processor = LogMelProcessor::new(config);
    frames
        .iter()
        .map(|frame| processor.process_frame(frame))
        .filter(|row| !row.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{log_compress, log_mel_features, log_mel_frame, LogMelConfig, LogMelProcessor};

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

    #[test]
    fn reusable_processor_matches_free_function() {
        let config = LogMelConfig::speech_default(16_000, 400);
        let frame: Vec<f32> = (0..400).map(|index| (index as f32 * 0.1).sin()).collect();
        let expected = log_mel_frame(&frame, config);
        let mut processor = LogMelProcessor::new(config);

        assert_eq!(processor.process_frame(&frame), expected);
        assert_eq!(processor.process_frame(&frame), expected);
    }

    #[test]
    fn reusable_processor_rejects_wrong_frame_size() {
        let mut processor = LogMelProcessor::new(LogMelConfig::speech_default(16_000, 400));

        assert!(processor.process_frame(&[0.0; 399]).is_empty());
    }
}
