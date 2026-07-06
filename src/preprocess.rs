#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreprocessConfig {
    pub target_sample_rate_hz: u32,
    pub target_channels: u16,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            target_sample_rate_hz: 16_000,
            target_channels: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedAudio {
    pub samples: Vec<f32>,
    pub sample_rate_hz: u32,
    pub channels: u16,
}

pub fn interleaved_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    match channels {
        0 => Vec::new(),
        1 => samples.to_vec(),
        channels => {
            let channels = channels as usize;
            samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32)
                .collect()
        }
    }
}

pub fn resample_linear(samples: &[f32], source_rate_hz: u32, target_rate_hz: u32) -> Vec<f32> {
    if samples.is_empty() || source_rate_hz == 0 || target_rate_hz == 0 {
        return Vec::new();
    }

    if source_rate_hz == target_rate_hz {
        return samples.to_vec();
    }

    let output_len =
        ((samples.len() as f64 * target_rate_hz as f64 / source_rate_hz as f64).round() as usize)
            .max(1);

    if output_len == 1 {
        return vec![samples[0]];
    }

    if samples.len() == 1 {
        return vec![samples[0]; output_len];
    }

    let last_input = samples.len() - 1;
    let last_output = output_len - 1;

    (0..output_len)
        .map(|i| {
            let position = i as f64 * last_input as f64 / last_output as f64;
            let left_index = position.floor() as usize;
            let right_index = position.ceil() as usize;
            if left_index == right_index {
                samples[left_index]
            } else {
                let fraction = (position - left_index as f64) as f32;
                let left = samples[left_index];
                let right = samples[right_index];
                left + (right - left) * fraction
            }
        })
        .collect()
}

pub fn preprocess_audio(
    samples: &[f32],
    source_sample_rate_hz: u32,
    source_channels: u16,
    config: PreprocessConfig,
) -> PreprocessedAudio {
    let mono = interleaved_to_mono(samples, source_channels);
    let resampled = resample_linear(&mono, source_sample_rate_hz, config.target_sample_rate_hz);

    PreprocessedAudio {
        samples: resampled,
        sample_rate_hz: config.target_sample_rate_hz,
        channels: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::{interleaved_to_mono, preprocess_audio, resample_linear, PreprocessConfig};
    use crate::{
        config::AppConfig, extractor::extract_log_mel_from_samples, streaming::StreamingExtractor,
    };

    #[test]
    fn preprocess_config_default_is_16khz_mono() {
        let config = PreprocessConfig::default();

        assert_eq!(config.target_sample_rate_hz, 16_000);
        assert_eq!(config.target_channels, 1);
    }

    #[test]
    fn interleaved_to_mono_empty_input() {
        assert!(interleaved_to_mono(&[], 2).is_empty());
    }

    #[test]
    fn interleaved_to_mono_zero_channels_returns_empty() {
        assert!(interleaved_to_mono(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn interleaved_to_mono_single_channel_passthrough() {
        assert_eq!(
            interleaved_to_mono(&[1.0, 2.0, 3.0], 1),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn interleaved_to_mono_stereo_average() {
        assert_eq!(
            interleaved_to_mono(&[1.0, 3.0, 2.0, 4.0], 2),
            vec![2.0, 3.0]
        );
    }

    #[test]
    fn interleaved_to_mono_ignores_incomplete_trailing_group() {
        assert_eq!(interleaved_to_mono(&[1.0, 3.0, 2.0], 2), vec![2.0]);
    }

    #[test]
    fn resample_linear_empty_input() {
        assert!(resample_linear(&[], 48_000, 16_000).is_empty());
    }

    #[test]
    fn resample_linear_same_rate_passthrough() {
        let samples = vec![1.0, 2.0, 3.0];

        assert_eq!(resample_linear(&samples, 16_000, 16_000), samples);
    }

    #[test]
    fn resample_linear_downsamples_length() {
        let samples = vec![1.0; 48_000];
        let resampled = resample_linear(&samples, 48_000, 16_000);

        assert_eq!(resampled.len(), 16_000);
    }

    #[test]
    fn resample_linear_upsamples_length() {
        let samples = vec![1.0; 16_000];
        let resampled = resample_linear(&samples, 16_000, 48_000);

        assert_eq!(resampled.len(), 48_000);
    }

    #[test]
    fn preprocess_audio_converts_stereo_48khz_to_mono_16khz() {
        let stereo_frames = 48_000;
        let mut samples = Vec::with_capacity(stereo_frames * 2);
        for _ in 0..stereo_frames {
            samples.push(1.0);
            samples.push(3.0);
        }

        let output = preprocess_audio(&samples, 48_000, 2, PreprocessConfig::default());

        assert_eq!(output.channels, 1);
        assert_eq!(output.sample_rate_hz, 16_000);
        assert_eq!(output.samples.len(), 16_000);
    }

    #[test]
    fn preprocess_audio_output_metadata_is_target() {
        let output = preprocess_audio(&[1.0; 48_000], 48_000, 1, PreprocessConfig::default());

        assert_eq!(output.sample_rate_hz, 16_000);
        assert_eq!(output.channels, 1);
    }

    #[test]
    fn preprocess_then_extract_has_expected_shape() {
        let output = preprocess_audio(&[0.25; 1_600], 16_000, 1, PreprocessConfig::default());
        let features = extract_log_mel_from_samples(&output.samples, &AppConfig::default());

        assert_eq!(features.num_frames, 8);
        assert_eq!(features.num_bins, 40);
    }

    #[test]
    fn preprocess_then_stream_in_hop_chunks_emits_expected_frames() {
        let output = preprocess_audio(&[0.25; 1_600], 16_000, 1, PreprocessConfig::default());
        let mut extractor = StreamingExtractor::new(AppConfig::default());

        for chunk in output.samples.chunks(160) {
            extractor.push_samples(chunk);
        }

        assert_eq!(extractor.total_emitted_frames(), 8);
    }

    #[test]
    fn stereo_48khz_preprocesses_and_extracts() {
        let samples: Vec<f32> = (0..48_000)
            .flat_map(|index| {
                let sample = (index as f32 * 0.01).sin();
                [sample, sample]
            })
            .collect();
        let output = preprocess_audio(&samples, 48_000, 2, PreprocessConfig::default());
        let features = extract_log_mel_from_samples(&output.samples, &AppConfig::default());

        assert_eq!(output.samples.len(), 16_000);
        assert!(features.num_frames > 0);
        assert_eq!(features.num_bins, 40);
    }
}
